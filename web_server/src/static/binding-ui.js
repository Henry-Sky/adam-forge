export function initializeBinding({projectId,initialJobId,requestPayload,isSaved,preview,notify,onAvailability}){
  const $=id=>document.getElementById(id),key='adam-skinning-job:'+projectId;
  let jobId,result,previewData,busy=false,timer,testing=false;
  const url=path=>path+(path.includes('?')?'&':'?')+'project='+encodeURIComponent(projectId);
  async function read(response){
    if(!response.headers.get('Content-Type')?.includes('application/json'))throw Error('蒙皮接口不可用，请重启网页服务后刷新');
    const value=await response.json();if(!response.ok)throw Error(value.error||'蒙皮请求失败');return value;
  }
  function updateDirty(){
    const saved=isSaved(),ready=result?.state==='ready'&&saved;
    $('bind-skinning').disabled=testing||busy||result?.state==='running'||!saved;
    for(const id of ['binding-view'])$(id).disabled=testing||busy||!ready;
    if(!saved&&$('binding-view').value!=='original'){$('binding-view').value='original';preview('original');}
    $('bind-skinning').title=saved?'从已保存模型生成 Gaussian 蒙皮绑定':'请先保存当前配准';
    const state=$('binding-state');
    state.textContent=testing?'LBS 测试中':ready?'已绑定 · '+result.report.deformation_joints+' 关节':!saved&&result?'绑定已失效':result?.state==='running'?`绑定中 · ${result.progress||0}%`:result?.state==='stale'?'绑定已失效':result?.state==='error'?'绑定失败':'未绑定';
    state.classList.toggle('bound',ready);state.classList.toggle('dirty',result?.state==='stale'||(!saved&&Boolean(result)));
    onAvailability?.(Boolean(ready&&!busy&&!testing),jobId);

  }
  async function poll(){
    if(!jobId)return;
    try{
      result=await read(await fetch(url('/api/bindings?job='+encodeURIComponent(jobId))));
      $('binding-status').textContent=result.message+(result.state==='running'?` · ${result.progress}%`:'');
      $('binding-progress').value=result.progress||0;
      $('binding-progress').hidden=result.state!=='running';
      $('binding-error').textContent=result.detail||'';$('binding-error-details').hidden=!result.detail;
      if(result.report){
        const r=result.report;
        $('binding-summary').textContent=`${r.gaussians.toLocaleString()} Gaussian · ${r.deformation_joints} 关节 · 每点 ${r.top_k} 骨骼\n距表面较远：${r.needs_review_count.toLocaleString()} 点 · 位置回放误差 ${r.identity_center_max_error.toExponential(2)}`;
      }
      updateDirty();
      if(result.state==='running')timer=setTimeout(poll,1500);
    }catch(error){$('binding-status').textContent=error.message;result={state:'error'};updateDirty();}
  }
  $('bind-skinning').onclick=async()=>{
    if(!isSaved())return notify('请先保存配准，再生成蒙皮绑定',true);
    busy=true;updateDirty();
    try{
      const value={...requestPayload(),top_k:Number($('binding-influences').value)};
      result=await read(await fetch(url('/api/bindings'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)}));
      jobId=result.id;previewData=undefined;preview('original');$('binding-view').value='original';
      try{localStorage.setItem(key,jobId);}catch{}
      clearTimeout(timer);await poll();
    }catch(error){notify(error.message,true);}finally{busy=false;updateDirty();}
  };
  $('binding-view').onchange=async()=>{
    const mode=$('binding-view').value;
    try{
      if(mode!=='original'&&!previewData){
        previewData=await read(await fetch(url(`/bindings/${jobId}/preview.json`)));
      }
      preview(mode,previewData);
    }catch(error){$('binding-view').value='original';preview('original');notify(error.message,true);}
  };
  jobId=initialJobId;
  try{jobId=initialJobId||localStorage.getItem(key);}catch{}
  if(jobId)poll();
  updateDirty();
  return {updateDirty,setTesting(value){testing=value;updateDirty();},getReadyBinding(){if(result?.state!=='ready'||!isSaved()||busy)throw Error('请先保存配准并完成蒙皮绑定');return jobId;}};
}
