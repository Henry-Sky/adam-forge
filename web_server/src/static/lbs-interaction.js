import * as THREE from 'three';
import {TransformControls} from 'three/addons/controls/TransformControls.js';
import {dyno} from '@sparkjsdev/spark';

// The same blend is used by real covariance splats and the point inspection view.
// Float weights retain every stored influence, including K=8/16/32 bindings.
export function lbsShader(influences,width){
  if(!Number.isInteger(influences)||influences<1||influences>32)throw Error('影响骨骼数不正确');
  return `mat4 adamLbsBlend(int splatIndex, sampler2D pairs, sampler2D bones) {
    mat4 blend = mat4(0.0);
    for (int k=0; k<${influences}; ++k) {
      int address = splatIndex*${influences}+k;
      vec2 pair = texelFetch(pairs, ivec2(address%${width},address/${width}), 0).rg;
      if (pair.y > 0.0) {
        int bone = int(pair.x+0.5);
        mat4 delta = mat4(texelFetch(bones,ivec2(0,bone),0),texelFetch(bones,ivec2(1,bone),0),texelFetch(bones,ivec2(2,bone),0),texelFetch(bones,ivec2(3,bone),0));
        blend += pair.y*delta;
      }
    }
    return blend;
  }`;
}

export function createLbsInteraction({metadata,pairs,scene,renderer,getCamera,getOrbit,splats,points,onSelect}){
  const {gaussians,influences,joint_names:names,joint_parent_ids:parents}=metadata;
  if(pairs.length!==gaussians*influences*2||gaussians!==points.geometry.attributes.position.count)throw Error('蒙皮数据与当前 Gaussian 不匹配');
  const width=Math.min(1024,renderer.capabilities.maxTextureSize),height=Math.ceil(gaussians*influences/width);
  if(height>renderer.capabilities.maxTextureSize)throw Error('蒙皮数据超过当前显卡纹理大小限制');
  const skinData=new Float32Array(width*height*2);skinData.set(pairs);
  const skinTexture=new THREE.DataTexture(skinData,width,height,THREE.RGFormat,THREE.FloatType);
  skinTexture.internalFormat='RG32F';skinTexture.needsUpdate=true;
  const boneData=new Float32Array(names.length*16);
  const boneTexture=new THREE.DataTexture(boneData,4,names.length,THREE.RGBAFormat,THREE.FloatType);
  boneTexture.internalFormat='RGBA32F';boneTexture.needsUpdate=true;
  const shader=lbsShader(influences,width);
  const modifier=new dyno.Dyno({
    inTypes:{covsplat:dyno.CovSplat,pairs:'sampler2D',bones:'sampler2D'},outTypes:{covsplat:dyno.CovSplat},
    inputs:{pairs:new dyno.DynoSampler2D({value:skinTexture}),bones:new dyno.DynoSampler2D({value:boneTexture})},
    globals:()=>[shader],
    statements:({inputs,outputs})=>{
      const s=outputs.covsplat;
      return `${s} = ${inputs.covsplat};
      if (isCovSplatActive(${s}.flags)) {
        mat4 blend = adamLbsBlend(${s}.index,${inputs.pairs},${inputs.bones});
        mat3 basis = mat3(blend);
        ${s}.center = (blend*vec4(${s}.center,1.0)).xyz;
        vec3 d = ${s}.xxyyzz, e = ${s}.xyxzyz;
        mat3 covariance = mat3(d.x,e.x,e.y,e.x,d.y,e.z,e.y,e.z,d.z);
        covariance = basis*covariance*transpose(basis);
        ${s}.xxyyzz = vec3(covariance[0][0],covariance[1][1],covariance[2][2]);
        ${s}.xyxzyz = vec3(covariance[0][1],covariance[0][2],covariance[1][2]);
      }`.split('\n');
    }
  });
  const originalModifiers=splats.covObjectModifiers,originalMaterial=points.material;
  splats.covObjectModifiers=[...(originalModifiers||[]),modifier];splats.updateGenerator();
  const skinIndex=new Float32Array(gaussians);for(let i=0;i<gaussians;i++)skinIndex[i]=i;
  points.geometry.setAttribute('skinIndex',new THREE.BufferAttribute(skinIndex,1));
  const pointMaterial=new THREE.ShaderMaterial({
    uniforms:{...originalMaterial.uniforms,adamPairs:{value:skinTexture},adamBones:{value:boneTexture}},
    vertexShader:`uniform sampler2D adamPairs;uniform sampler2D adamBones;
      attribute float skinIndex;attribute vec3 color;attribute float alpha;varying vec3 c;varying float a;
      ${shader}
      void main(){c=color;a=alpha;vec3 moved=(adamLbsBlend(int(skinIndex+0.5),adamPairs,adamBones)*vec4(position,1.0)).xyz;
      gl_Position=projectionMatrix*modelViewMatrix*vec4(moved,1.0);gl_PointSize=2.3;}`,
    fragmentShader:originalMaterial.fragmentShader
  });
  points.material=pointMaterial;points.frustumCulled=false;
  const matrix=value=>new THREE.Matrix4().set(...value.flat());
  const restWorld=metadata.bind_world.map(matrix),inverseBind=restWorld.map(m=>m.clone().invert());
  const nodes=names.map(()=>new THREE.Object3D()),roots=[];
  const initial=[];
  nodes.forEach((node,i)=>{
    matrix(metadata.bind_local[i]).decompose(node.position,node.quaternion,node.scale);
    initial.push({position:node.position.clone(),quaternion:node.quaternion.clone(),scale:node.scale.clone()});
    if(parents[i]>=0&&parents[i]!==i)nodes[parents[i]].add(node);else{scene.add(node);roots.push(node);}
  });
  const maxZ=Math.max(...restWorld.map(m=>m.elements[14])),minZ=Math.min(...restWorld.map(m=>m.elements[14]));
  const radius=Math.max((maxZ-minZ)*.007,.003);
  const markerGeometry=new THREE.SphereGeometry(radius,8,6);
  const markerMaterial=new THREE.MeshBasicMaterial({color:0xc5ff91,depthTest:false});
  const selectedMaterial=new THREE.MeshBasicMaterial({color:0xffd27c,depthTest:false});
  const markers=nodes.map((node,i)=>{
    const marker=new THREE.Mesh(markerGeometry,markerMaterial);marker.renderOrder=40;marker.userData.jointIndex=i;node.add(marker);return marker;
  });
  const lineGeometry=new THREE.BufferGeometry();
  const linePositions=new Float32Array(parents.filter((p,i)=>p>=0&&p!==i).length*6);
  lineGeometry.setAttribute('position',new THREE.BufferAttribute(linePositions,3));
  const lineMaterial=new THREE.LineBasicMaterial({color:0xc5ff91,depthTest:false});
  const lines=new THREE.LineSegments(lineGeometry,lineMaterial);lines.renderOrder=39;lines.frustumCulled=false;scene.add(lines);
  const control=new TransformControls(getCamera(),renderer.domElement);control.setMode('rotate');control.setSpace('local');control.setSize(.8);
  const helper=control.getHelper();scene.add(helper);
  let selected=-1,disposed=false;
  const delta=new THREE.Matrix4(),v=new THREE.Vector3();
  function update(){
    roots.forEach(root=>root.updateMatrixWorld(true));
    nodes.forEach((node,i)=>{delta.multiplyMatrices(node.matrixWorld,inverseBind[i]);boneData.set(delta.elements,i*16);});
    boneTexture.needsUpdate=true;splats.updateVersion();
    let offset=0;
    nodes.forEach((node,i)=>{if(parents[i]<0||parents[i]===i)return;for(const item of [node,nodes[parents[i]]]){item.getWorldPosition(v);linePositions.set(v.toArray(),offset);offset+=3;}});
    lineGeometry.attributes.position.needsUpdate=true;lineGeometry.computeBoundingSphere();
  }
  control.addEventListener('objectChange',update);
  control.addEventListener('dragging-changed',event=>{getOrbit().enabled=!event.value;});
  function select(index){
    if(!Number.isInteger(index)||index<0||index>=nodes.length)return;
    selected=index;markers.forEach((marker,i)=>marker.material=i===selected?selectedMaterial:markerMaterial);
    control.attach(nodes[index]);onSelect?.(index,names[index]);
  }
  const raycaster=new THREE.Raycaster(),mouse=new THREE.Vector2();let pointerDown;
  function down(event){pointerDown=[event.clientX,event.clientY];}
  function up(event){
    if(event.button!==0||!pointerDown||Math.hypot(event.clientX-pointerDown[0],event.clientY-pointerDown[1])>4)return;
    const rect=renderer.domElement.getBoundingClientRect();
    mouse.set((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);
    raycaster.setFromCamera(mouse,getCamera());
    const hit=raycaster.intersectObjects(markers.filter(marker=>marker.visible),false)[0];
    if(hit)select(hit.object.userData.jointIndex);
  }
  renderer.domElement.addEventListener('pointerdown',down);renderer.domElement.addEventListener('pointerup',up);
  update();select(Math.max(0,names.indexOf('LeftForeArm')));
  return {
    nodes,control,boneData,skinTexture,boneTexture,shader,update,select,
    setCamera(camera){control.camera=camera;},
    setBoneVisibility(visible){markers.forEach(marker=>marker.visible=visible);lines.visible=visible;},
    reset(){nodes.forEach((node,i)=>{node.position.copy(initial[i].position);node.quaternion.copy(initial[i].quaternion);node.scale.copy(initial[i].scale);});update();},
    dispose(){
      if(disposed)return;disposed=true;control.detach();control.dispose();scene.remove(helper);
      renderer.domElement.removeEventListener('pointerdown',down);renderer.domElement.removeEventListener('pointerup',up);
      roots.forEach(root=>scene.remove(root));scene.remove(lines);
      lineGeometry.dispose();lineMaterial.dispose();markerGeometry.dispose();markerMaterial.dispose();selectedMaterial.dispose();
      splats.covObjectModifiers=originalModifiers;splats.updateGenerator();
      points.material=originalMaterial;points.geometry.deleteAttribute('skinIndex');points.frustumCulled=true;
      pointMaterial.dispose();skinTexture.dispose();boneTexture.dispose();getOrbit().enabled=true;
    }
  };
}
