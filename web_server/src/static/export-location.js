import {chooseServerDirectory} from './asset-import.js';
// Browser-owned directory handles never become server filesystem paths.
const key='adam-registration-export';
let handle;
const storage={get(k){try{return localStorage.getItem(key+k);}catch{return null;}},set(k,v){try{localStorage.setItem(key+k,v);}catch{}}};
async function database(){return new Promise((resolve,reject)=>{const req=indexedDB.open(key,1);req.onupgradeneeded=()=>req.result.createObjectStore('handles');req.onsuccess=()=>resolve(req.result);req.onerror=()=>reject(req.error);});}
async function handleStore(value){
  const db=await database();
  try{return await new Promise((resolve,reject)=>{const transaction=db.transaction('handles',value?'readwrite':'readonly');const request=value?transaction.objectStore('handles').put(value,'directory'):transaction.objectStore('handles').get('directory');let result;request.onsuccess=()=>{result=request.result;};transaction.oncomplete=()=>resolve(result);transaction.onerror=()=>reject(transaction.error);});}finally{db.close();}
}
export async function initializeExportLocation(){
  try{handle=await handleStore();}catch{/* Directory memory is optional. */}
}
function serverDirectory(){
  const dialog=document.createElement('dialog');dialog.className='project-form asset-dialog';
  dialog.innerHTML='<form><div class="heading-row"><h2>导出到服务器</h2><button type="button" id="close-server-export" aria-label="关闭导出窗口">×</button></div><p class="caption">导出配准后的 SOMA 与 Gaussian，自动创建独立子目录。</p><label>导出目录<div class="server-path-row"><input id="server-export-path" required placeholder="/服务器/导出根目录"><button type="button" id="browse-export-directory">浏览…</button></div></label><p id="server-export-message" class="caption" role="status">记住上次选择的目录，不覆盖已有导出文件。</p><button type="submit" class="primary full-width">确认导出</button></form>';
  const input=dialog.querySelector('input');input.value=storage.get('-server')||'';
  document.body.append(dialog);
  return new Promise((resolve,reject)=>{
    let settled=false;
    dialog.querySelector('form').onsubmit=event=>{event.preventDefault();const path=input.value.trim();if(!path)return;settled=true;storage.set('-server',path);dialog.close();dialog.remove();resolve(path);};
    dialog.querySelector('#close-server-export').onclick=()=>dialog.close();
    dialog.querySelector('#browse-export-directory').onclick=async()=>{
      try{input.value=await chooseServerDirectory(input.value);}
      catch(error){if(error.name!=='AbortError')dialog.querySelector('#server-export-message').textContent=error.message;}
    };
    dialog.onclose=()=>{dialog.remove();if(!settled)reject(new DOMException('已取消导出','AbortError'));};
    dialog.showModal();input.focus();
  });
}
export async function prepareExport(destination='server'){
  storage.set('-mode',destination);
  if(destination==='server')return {destination,export_directory:await serverDirectory()};
  if(window.showDirectoryPicker){
    const options={id:'soma-registration-export',mode:'readwrite'};if(handle)options.startIn=handle;
    handle=await window.showDirectoryPicker(options);
    if(await handle.queryPermission({mode:'readwrite'})!=='granted'&&await handle.requestPermission({mode:'readwrite'})!=='granted')throw Error('未获得本地目录写入权限');
    try{await handleStore(handle);}catch{}
  }
  return {destination};
}
export async function finishLocalExport(result){
  if(!handle||!window.showDirectoryPicker){
    const a=document.createElement('a');a.href=`/exports/${result.export_id}/download.zip`;a.download='aligned-soma-gaussians.zip';a.click();return 'ZIP 下载';
  }
  const name='aligned-'+new Date().toISOString().replace(/[:.]/g,'-')+'-'+result.export_id.slice(0,6);
  const directory=await handle.getDirectoryHandle(name,{create:true});
  try{
    for(const file of result.files){
      const response=await fetch(`/exports/${result.export_id}/${file}`);if(!response.ok)throw Error('下载导出文件失败：'+file);
      const target=await directory.getFileHandle(file,{create:true}),stream=await target.createWritable();
      try{await stream.write(await response.blob());await stream.close();}catch(error){await stream.abort();throw error;}
    }
  }catch(error){try{await handle.removeEntry(name,{recursive:true});}catch{}throw error;}
  return handle.name+'/'+name;
}
