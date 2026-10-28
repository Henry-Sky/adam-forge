import {initializeExportLocation,prepareExport,finishLocalExport} from './export-location.js';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { validateMatrix, pivotMatrix, sourceMatrix } from './math.js';

const projectId=new URLSearchParams(location.search).get('project');
const assetUrl=path=>projectId?path+'?project='+encodeURIComponent(projectId):path;
const $=id=>document.getElementById(id);
let data, renderer, scene, camera, orbit, gizmo, rig, mesh, bones, points, splats, grid;
let currentView='perspective', baseMatrix, undoStack=[], redoStack=[], savedMatrix, loading=true;
let cloudVisible=true, splatsReady=false, draftKey, restoreDraft=false;
// Right-handed basis: old (x,y,z) -> new (z,x,y); new world is Z-up.
const displayBasis=new THREE.Matrix4().set(0,0,1,0, 1,0,0,0, 0,1,0,0, 0,0,0,1);
const displayRotation=new THREE.Quaternion().setFromRotationMatrix(displayBasis);
let notificationTimer, cloudRoot, savedViewerState;
function hasChanges(){return data?.project_mode&&savedViewerState?!sameState(snapshot(),savedViewerState):!equal(sourceTransform(),savedMatrix);}
function activeObject(){return $("edit-target").value==="gaussian"?cloudRoot:rig;}
function snapshot(){rig.updateMatrix();cloudRoot.updateMatrix();return {soma:rowMatrix(rig.matrix),cloud:rowMatrix(cloudRoot.matrix)};}
function restore(state){threeMatrix(state.soma).decompose(rig.position,rig.quaternion,rig.scale);threeMatrix(state.cloud).decompose(cloudRoot.position,cloudRoot.quaternion,cloudRoot.scale);syncInputs();}
function sameState(a,b){return a&&b&&equal(a.soma,b.soma)&&equal(a.cloud,b.cloud);}
const viewport=$('viewport');
const numeric={position:[],rotation:[]};
const rowMatrix=m=>Array.from({length:4},(_,r)=>Array.from({length:4},(_,c)=>m.elements[c*4+r]));
const threeMatrix=m=>new THREE.Matrix4().set(...m.flat());
const equal=(a,b)=>a&&b&&a.flat().every((v,i)=>Math.abs(v-b.flat()[i])<1e-9);

function notify(message,error=false){
  $('notification').textContent=message;
  $('notification').classList.toggle('error',error);
  $('notification').style.display='block';
  clearTimeout(notificationTimer);
  notificationTimer=setTimeout(()=>$('notification').style.display='none',error?9000:4500);
}
function sourceTransform(){
  rig.updateMatrix();cloudRoot.updateMatrix();
  const relative=cloudRoot.matrix.clone().invert().multiply(rig.matrix);
  return sourceMatrix(rowMatrix(relative),data.pivot);
}
function setTransform(matrix){
  validateMatrix(matrix);
  cloudRoot.updateMatrix();
  cloudRoot.matrix.clone().multiply(threeMatrix(pivotMatrix(matrix,data.pivot))).decompose(rig.position,rig.quaternion,rig.scale);
  syncInputs();
}
function commit(){
  const m=sourceTransform();
  const state=snapshot();
  if(sameState(state,undoStack.at(-1)))return;
  undoStack.push(state);if(undoStack.length>100)undoStack.shift();redoStack=[];
  rememberDraft(m);syncInputs();
}
function rememberDraft(m){
  try{localStorage.setItem(draftKey,JSON.stringify(payload(m)));}catch{/* Disk save remains available. */}
}
function payload(m=sourceTransform()){
  return {world_axes:'X-front Y-left Z-up',revision:data.revision,format:'soma_manual_registration_v1',source_ply_sha256:data.source_ply_sha256,
    source_soma_sha256:data.source_soma_sha256,soma_camera_to_ply:m,
    coordinates:'column vectors: ply = M @ [soma_camera_metres,1]',
    viewer_state:snapshot(),scope:'global similarity only; matrix maps into ORIGINAL PLY coordinates',weights_status:'recompute skinning weights after registration'};
}
function syncInputs(){
  if(!rig)return;
  const object=activeObject()||rig;
  const angles=new THREE.Euler().setFromQuaternion(object.quaternion,'XYZ');
  numeric.position.forEach((el,i)=>el.value=object.position.getComponent(i).toFixed(5));
  numeric.rotation.forEach((el,i)=>el.value=THREE.MathUtils.radToDeg(angles.toArray()[i]).toFixed(3));
  $('scale').value=object.scale.x.toFixed(6);
  const m=sourceTransform();
  $('matrix').textContent=m.map(row=>row.map(v=>(v>=0?' ':'')+v.toFixed(5)).join('  ')).join('\n');
  $('undo').disabled=undoStack.length<2;$('redo').disabled=!redoStack.length;
  const dirty=hasChanges();
  $('save-state').textContent=dirty?'待导出':'已导出 / 初始状态';
  $('save-state').classList.toggle('dirty',dirty);
}
for(const kind of ['position','rotation']){
  for(let i=0;i<3;i++){
    const label=document.createElement('label');
    const axis=document.createElement('span');axis.className=['x','y','z'][i];axis.textContent=['X','Y','Z'][i];
    const input=document.createElement('input');input.type='number';input.step=kind==='position'?'0.001':'0.1';input.value='0';
    input.setAttribute('aria-label',`${kind==='position'?'位置':'旋转'} ${axis.textContent}`);
    label.append(axis,input);$(kind+'-inputs').append(label);numeric[kind].push(input);
    input.addEventListener('change',()=>{
      if(!rig)return;
      const values=numeric[kind].map(x=>Number(x.value));
      if(!values.every(Number.isFinite)){syncInputs();return notify('请输入有限数值',true);}
      const object=activeObject();
      if(kind==='position')object.position.fromArray(values);
      else object.quaternion.setFromEuler(new THREE.Euler(...values.map(THREE.MathUtils.degToRad),'XYZ'));
      commit();
    });
  }
}
function setScale(s){
  if(!rig)return;
  if(!Number.isFinite(s)||s<=1e-6||s>=1e6){syncInputs();return notify('缩放必须为正且在有效范围内',true);}
  activeObject().scale.setScalar(s);commit();
}
$('scale').addEventListener('change',e=>setScale(Number(e.target.value)));
$('scale-up').onclick=()=>rig&&setScale(activeObject().scale.x*1.01);
$('scale-down').onclick=()=>rig&&setScale(activeObject().scale.x/1.01);
function mode(value){
  if(!gizmo)return;
  gizmo.setMode(value);
  for(const name of ['translate','rotate'])$(name).classList.toggle('selected',name===value);
}
$('translate').onclick=()=>mode('translate');$('rotate').onclick=()=>mode('rotate');
$('space').onchange=e=>gizmo?.setSpace(e.target.value);
$('undo').onclick=()=>{if(undoStack.length<2)return;redoStack.push(undoStack.pop());restore(undoStack.at(-1));rememberDraft(sourceTransform());};
$('redo').onclick=()=>{if(!redoStack.length)return;const m=redoStack.pop();undoStack.push(m);restore(m);rememberDraft(sourceTransform());};
$('reset').onclick=()=>{if(!rig)return;cloudRoot.position.set(0,0,0);cloudRoot.quaternion.copy(displayRotation);cloudRoot.scale.setScalar(1);setTransform(baseMatrix);commit();notify('已恢复自动配准结果，可撤销');};
$('show-cloud').onchange=e=>{cloudVisible=e.target.checked;showCloud();};
$('show-bones').onchange=e=>{if(bones)bones.visible=e.target.checked;};
$('show-mesh').onchange=e=>{if(mesh)mesh.visible=e.target.checked;};
$('wireframe').onchange=e=>{if(mesh)mesh.material.wireframe=e.target.checked;};
$('mesh-alpha').oninput=e=>{if(mesh)mesh.material.opacity=Number(e.target.value);$('mesh-alpha-value').textContent=Math.round(Number(e.target.value)*100)+'%';};
$('grid-toggle').onclick=()=>{if(!grid)return;grid.visible=!grid.visible;$('grid-toggle').classList.toggle('selected',grid.visible);};
$('toggle-reference').onclick=()=>{const hidden=$('reference').hidden=!$('reference').hidden;$('toggle-reference').textContent=hidden?'展开':'收起';};
$('threshold').oninput=e=>{$('threshold-value').textContent=Number(e.target.value).toFixed(2);if(points)points.material.uniforms.threshold.value=Number(e.target.value);};
$('cloud-mode').onchange=()=>showCloud();
function showCloud(){
  const useSplats=$('cloud-mode').value==='splats'&&splatsReady;
  if(splats)splats.visible=cloudVisible&&useSplats;
  if(points)points.visible=cloudVisible&&!useSplats;
  if(points)$('render-status').textContent=useSplats?'Gaussian splats · WebGL2':'点云检查模式 · Gaussian 中心';
  $('threshold').disabled=useSplats;
  $('threshold').title=useSplats?'阈值仅用于点云检查模式':'';
}
const target=new THREE.Vector3();
function drawAxisPreview(){
  if(!camera)return;
  const canvas=$('axis-preview'),ratio=Math.min(devicePixelRatio||1,2),size=116;
  if(canvas.width!==size*ratio){canvas.width=size*ratio;canvas.height=size*ratio;}
  const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,size,size);
  ctx.beginPath();ctx.arc(58,58,53,0,Math.PI*2);ctx.fillStyle='#101820c9';ctx.fill();ctx.strokeStyle='#344352';ctx.stroke();
  const inverse=camera.quaternion.clone().invert();
  const axes=[{v:[1,0,0],name:'X',color:'#f08d89'},{v:[0,1,0],name:'Y',color:'#b9ed8a'},{v:[0,0,1],name:'Z',color:'#85bfff'}]
    .map(axis=>({...axis,p:new THREE.Vector3(...axis.v).applyQuaternion(inverse)})).sort((a,b)=>a.p.z-b.p.z);
  for(const {p,name,color} of axes){
    const x=p.x*33,y=-p.y*33;
    ctx.strokeStyle=color;ctx.globalAlpha=.35;ctx.lineWidth=1;ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(58,58);ctx.lineTo(58-x,58-y);ctx.stroke();ctx.setLineDash([]);
    ctx.globalAlpha=p.z<-.01?.6:1;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(58,58);ctx.lineTo(58+x,58+y);ctx.stroke();
    const endX=58+p.x*42,endY=58-p.y*42;
    ctx.beginPath();ctx.arc(endX,endY,11,0,Math.PI*2);ctx.fillStyle='#17222d';ctx.fill();ctx.strokeStyle=color;ctx.lineWidth=1.2;ctx.stroke();
    ctx.fillStyle=color;ctx.font='bold 10px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('+'+name,endX,endY);
  }
  ctx.globalAlpha=1;ctx.beginPath();ctx.arc(58,58,3,0,Math.PI*2);ctx.fillStyle='#d6e4ee';ctx.fill();
}
function selectView(view){
  if(!renderer)return;
  const width=viewport.clientWidth,height=viewport.clientHeight,aspect=width/height;
  const old=orbit;
  const extent=1.4;
  if(view==='perspective')camera=new THREE.PerspectiveCamera(40,aspect,.001,100);
  else camera=new THREE.OrthographicCamera(-extent*aspect/2,extent*aspect/2,extent/2,-extent/2,.001,100);
  camera.up.set(...(view==='top'?[1,0,0]:[0,0,1]));
  const offset={perspective:[1.8,1.25,.5],front:[2.5,0,0],side:[0,2.5,0],back:[-2.5,0,0],top:[0,0,2.5]}[view];
  camera.position.copy(target).add(new THREE.Vector3(...offset));camera.lookAt(target);
  if(old)old.dispose();
  orbit=new OrbitControls(camera,renderer.domElement);orbit.target.copy(target);orbit.enableDamping=true;
  orbit.minDistance=.1;orbit.maxDistance=20;
  if(gizmo)gizmo.camera=camera;
  currentView=view;
  document.querySelectorAll('[data-view]').forEach(el=>el.classList.toggle('selected',el.dataset.view===view));
  $('view-label').textContent={perspective:'透视视图',front:'正交 · 正面 +X',side:'正交 · 左侧 +Y',back:'正交 · 背面 −X',top:'正交 · 顶视 +Z'}[view];
}
document.querySelectorAll('[data-view]').forEach(el=>el.onclick=()=>selectView(el.dataset.view));
$('fit').onclick=()=>selectView(currentView);
function resize(){
  if(!renderer||!camera)return;
  const w=viewport.clientWidth,h=viewport.clientHeight;
  renderer.setSize(w,h);
  if(camera.isPerspectiveCamera)camera.aspect=w/h;
  else{const height=camera.top-camera.bottom;camera.left=-height*w/h/2;camera.right=height*w/h/2;}
  camera.updateProjectionMatrix();
}
function bonesGroup(){
  const group=new THREE.Group();
  const positions=[],pivot=new THREE.Vector3(...data.pivot);
  const jointGeometry=new THREE.SphereGeometry(.007,8,6);
  const material=new THREE.MeshBasicMaterial({color:0xc5ff91,depthTest:false,transparent:true,opacity:.95});
  data.joints.forEach((j,i)=>{
    const v=new THREE.Vector3(...j).sub(pivot);
    const sphere=new THREE.Mesh(jointGeometry,material);sphere.position.copy(v);sphere.renderOrder=30;group.add(sphere);
    const parent=data.parents[i];if(parent>=0)positions.push(...v.toArray(),...new THREE.Vector3(...data.joints[parent]).sub(pivot).toArray());
  });
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
  const lines=new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({color:0xc5ff91,depthTest:false,transparent:true,opacity:1}));lines.renderOrder=31;group.add(lines);
  return group;
}
async function loadPoints(){
  const response=await fetch(assetUrl('/assets/gaussians.ply'));if(!response.ok)throw Error('PLY 读取失败');
  const buffer=await response.arrayBuffer();
  const geometry=new PLYLoader().parse(buffer);
  const count=geometry.attributes.position.count;
  // PLYLoader recognizes geometry only; read Gaussian DC / logit fields explicitly.
  const bytes=new Uint8Array(buffer);const endPattern=new TextEncoder().encode('end_header\n');let end=-1;
  outer:for(let i=0;i<Math.min(bytes.length,65536)-endPattern.length;i++){
    for(let j=0;j<endPattern.length;j++)if(bytes[i+j]!==endPattern[j])continue outer;
    end=i+endPattern.length;break;
  }
  if(end<0)throw Error('不支持的 PLY header');
  const header=new TextDecoder().decode(bytes.slice(0,end));
  if(!header.includes('format binary_little_endian'))throw Error('需要 binary little-endian Gaussian PLY');
  const props=header.split('\n').filter(x=>x.startsWith('property ')).map(x=>x.trim().split(/\s+/));
  if(props.some(x=>x[1]!=='float'))throw Error('本项目需要 Harvester float32 PLY');
  const index=name=>props.findIndex(x=>x[2]===name);const indices=['f_dc_0','f_dc_1','f_dc_2','opacity'].map(index);
  if(indices.some(i=>i<0))throw Error('PLY 缺少 Gaussian 颜色或 opacity 字段');
  const view=new DataView(buffer),stride=props.length*4,colors=new Float32Array(count*3),alphas=new Float32Array(count);
  for(let i=0;i<count;i++){
    for(let c=0;c<3;c++){const srgb=THREE.MathUtils.clamp(.5+.28209479177387814*view.getFloat32(end+i*stride+indices[c]*4,true),0,1);colors[3*i+c]=srgb<=.04045?srgb/12.92:((srgb+.055)/1.055)**2.4;}
    alphas[i]=1/(1+Math.exp(-view.getFloat32(end+i*stride+indices[3]*4,true)));
  }
  geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));geometry.setAttribute('alpha',new THREE.BufferAttribute(alphas,1));
  const material=new THREE.ShaderMaterial({uniforms:{threshold:{value:.1}},vertexShader:`attribute vec3 color;attribute float alpha;varying vec3 c;varying float a;void main(){c=color;a=alpha;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);gl_PointSize=2.3;}`,
    fragmentShader:`uniform float threshold;varying vec3 c;varying float a;void main(){if(a<threshold||length(gl_PointCoord-.5)>.5)discard;gl_FragColor=vec4(c,1.);\n#include <tonemapping_fragment>\n#include <colorspace_fragment>\n}`});
  points=new THREE.Points(geometry,material);cloudRoot.add(points);
  geometry.computeBoundingBox();const box=geometry.boundingBox;box.getCenter(target);
  const displayBox=box.clone().applyMatrix4(displayBasis);displayBox.getCenter(target);
  grid.position.z=displayBox.min.z-.01;
  return buffer;
}
async function loadSplats(buffer){
  try{
    const {SplatMesh,SparkRenderer}=await import('@sparkjsdev/spark');
    scene.add(new SparkRenderer({renderer}));
    splats=new SplatMesh({fileBytes:buffer,fileName:'gaussians.ply',lod:false});
    await splats.initialized;
    cloudRoot.add(splats);splatsReady=true;showCloud();
  }catch(error){
    console.error(error);$('cloud-mode').value='points';showCloud();notify('Gaussian 渲染加载失败，已切换点云检查模式：'+error.message,true);
  }
}
function initializeEmpty(){
  renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  viewport.prepend(renderer.domElement);scene=new THREE.Scene();scene.background=new THREE.Color(0x141b24);
  grid=new THREE.GridHelper(3,30,0x43514d,0x26313c);grid.rotation.x=Math.PI/2;scene.add(grid);selectView('perspective');resize();
  new ResizeObserver(resize).observe(viewport);
  renderer.setAnimationLoop(()=>{orbit.update();renderer.render(scene,camera);drawAxisPreview();});
  $('loading').style.display='none';$('save-state').textContent='空白资产';
  $('render-status').textContent='请从左侧导入或上传 SOMA 与 Gaussian';
  $('selection-status').textContent='尚未加载资产';
  document.querySelector('.reference-section').hidden=true;
  for(const id of ['save','download-models','reset','scale-up','scale-down'])$(id).disabled=true;
  window.registrationApp={ready:true,empty:true,renderer,scene};
}
async function initialize(){
  if(!projectId)return initializeEmpty();
  const response=await fetch(assetUrl('/api/scene'));if(!response.ok){initializeEmpty();notify('临时资产已失效，请重新导入 SOMA 和 PLY');return;}data=await response.json();
  baseMatrix=validateMatrix(data.matrix);savedMatrix=baseMatrix;
  if(data.project_mode){$('reset').textContent='恢复本次打开状态';document.querySelector('.local-note').lastChild.textContent=' 临时导入 · 导出到新目录 · 源文件不变';}
  draftKey=`soma-registration:zup-v1:${data.source_ply_sha256}:${data.source_soma_sha256}`;
  $('asset-name').textContent='Gaussian 模型';$('asset-count').textContent=data.gaussians.toLocaleString()+' Gaussian splats';
  $('soma-count').textContent=data.joints.length+' joints · '+(data.vertices.length/3).toLocaleString()+' vertices';
  for(const kind of ['gaussian','soma']){const card=$(kind+'-asset-card');card.classList.add('is-loaded');$(kind+'-asset-status').textContent='已导入';}
  if(data.reference)$('reference').src=assetUrl(data.reference);else document.querySelector('.reference-section').hidden=true;
  renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  renderer.outputColorSpace=THREE.SRGBColorSpace;viewport.prepend(renderer.domElement);
  scene=new THREE.Scene();scene.background=new THREE.Color(0x141b24);
  scene.add(new THREE.HemisphereLight(0xe0f2ff,0x243528,2));const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(2,3,4);scene.add(light);
  grid=new THREE.GridHelper(3,30,0x43514d,0x26313c);grid.rotation.x=Math.PI/2;scene.add(grid);
  cloudRoot=new THREE.Group();cloudRoot.quaternion.copy(displayRotation);scene.add(cloudRoot);
  rig=new THREE.Group();scene.add(rig);
  const vertices=new Float32Array(data.vertices);for(let i=0;i<vertices.length;i++)vertices[i]-=data.pivot[i%3];
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(vertices,3));geometry.setIndex(data.faces);geometry.computeVertexNormals();
  mesh=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:0x52bbd3,transparent:true,opacity:.3,side:THREE.DoubleSide,depthWrite:false,roughness:.8}));mesh.renderOrder=10;rig.add(mesh);
  bones=bonesGroup();rig.add(bones);setTransform(baseMatrix);
  const buffer=await loadPoints();selectView('perspective');
  gizmo=new TransformControls(camera,renderer.domElement);gizmo.attach(rig);scene.add(gizmo.getHelper());gizmo.setSize(.85);
  gizmo.addEventListener('dragging-changed',event=>{orbit.enabled=!event.value;if(!event.value)commit();});
  gizmo.addEventListener('objectChange',syncInputs);
  undoStack=[snapshot()];savedViewerState=snapshot();
  try{
    const draft=JSON.parse(localStorage.getItem(draftKey));
    if(draft&&((data.project_mode&&draft.viewer_state)?!sameState(draft.viewer_state,savedViewerState):!equal(draft.soma_camera_to_ply,baseMatrix))&&confirm('发现该资产的本地配准草稿，是否恢复？')){
      validateMatrix(draft.soma_camera_to_ply);if(draft.viewer_state){validateMatrix(draft.viewer_state.soma);validateMatrix(draft.viewer_state.cloud);restore(draft.viewer_state);}else setTransform(draft.soma_camera_to_ply);undoStack.push(snapshot());restoreDraft=true;
    }
  }catch(error){console.warn('Draft not restored',error);}
  syncInputs();resize();new ResizeObserver(resize).observe(viewport);
  loading=false;$('loading').hidden=true;$('loading').style.display='none';showCloud();
  let last=performance.now(),frames=0;
  renderer.setAnimationLoop(()=>{orbit.update();renderer.render(scene,camera);drawAxisPreview();frames++;const now=performance.now();if(now-last>1000){$('fps').textContent=Math.round(frames*1000/(now-last))+' FPS';frames=0;last=now;}});
  await loadSplats(buffer);
  if(restoreDraft)notify('已恢复浏览器本地草稿');
  window.registrationApp={getMatrix:sourceTransform,setMatrix:m=>{setTransform(m);commit();},ready:true,scene,rig,cloudRoot,renderer,
    renderOnce:()=>{orbit.update();renderer.render(scene,camera);drawAxisPreview();}};
}
$('copy-matrix').onclick=async()=>{if(!rig)return;try{await navigator.clipboard.writeText(JSON.stringify(sourceTransform()));notify('矩阵已复制');}catch{notify('浏览器不允许复制矩阵',true);}};
async function exportModels(target){
  if(!rig)return;$('save').disabled=true;$('download-models').disabled=true;
  try{
    const destination=await prepareExport(target);
    const p={...payload(),...destination};const response=await fetch(assetUrl('/api/save'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});
    const result=await response.json();if(!response.ok)throw Error(result.error||'保存失败');
    const location=result.destination==='local'?await finishLocalExport(result):result.directory;
    savedMatrix=p.soma_camera_to_ply;savedViewerState=snapshot();syncInputs();$('last-save').textContent='已导出：'+location;notify('已导出配准 SOMA 和 PLY；导入文件保持不变');
  }catch(error){if(error.name!=='AbortError')notify(error.message,true);}finally{$('save').disabled=false;$('download-models').disabled=false;}
}
$('save').onclick=()=>exportModels('server');
$('download-models').onclick=()=>exportModels('local');
window.addEventListener('keydown',e=>{
  if(loading||['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName))return;
  const k=e.key.toLowerCase();if((e.ctrlKey||e.metaKey)&&k==='s'){e.preventDefault();$('save').click();}
  else if((e.ctrlKey||e.metaKey)&&k==='z'){e.preventDefault();$(e.shiftKey?'redo':'undo').click();}
  else if((e.ctrlKey||e.metaKey)&&k==='y'){e.preventDefault();$('redo').click();}
  else if(k==='w')mode('translate');else if(k==='e')mode('rotate');else if(k==='f')$('fit').click();
});
window.addEventListener('beforeunload',e=>{if(rig&&hasChanges()){e.preventDefault();e.returnValue='';}});
window.addEventListener('error',e=>notify('页面错误：'+e.message,true));
initializeExportLocation().catch(console.warn);
initialize().catch(error=>{console.error(error);$('loading').innerHTML='<strong>场景加载失败</strong><span></span>';$('loading').querySelector('span').textContent=error.message;notify(error.message,true);});

$('edit-target').onchange=()=>{if(!gizmo)return;gizmo.attach(activeObject());syncInputs();$('selection-status').textContent=$('edit-target').value==='gaussian'?'正在调整高斯 · 原始 PLY 不修改':'正在调整 SOMA · 髋部操作中心';};
