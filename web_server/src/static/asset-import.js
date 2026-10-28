const $=id=>document.getElementById(id);
let mode='server';
function open(value){
  mode=value;const upload=value==='upload';
  $('asset-dialog-title').textContent=upload?'从本地上传':'从服务器导入';
  for(const kind of ['soma','ply']){
    $('asset-'+kind+'-server-row').hidden=upload;$('asset-'+kind+'-file').hidden=!upload;
  }
  $('asset-import-message').textContent='';$('asset-dialog').showModal();
}
$('import-assets').onclick=()=>open('server');$('upload-assets').onclick=()=>open('upload');
$('close-asset-dialog').onclick=()=>$('asset-dialog').close();
async function source(kind){
  if(mode==='server'){
    const path=$('asset-'+kind+'-path').value.trim();if(!path)throw Error('请填写 '+kind.toUpperCase()+' 服务器路径');
    return {mode,path};
  }
  const file=$('asset-'+kind+'-file').files[0];if(!file)throw Error('请选择 '+kind.toUpperCase()+' 文件');
  if(file.size>45*1024*1024)throw Error('本地单文件上限 45 MB，更大文件请先传到服务器');
  const data=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=()=>reject(Error('文件读取失败'));reader.readAsDataURL(file);});
  return {mode:'upload',data};
}
$('asset-form').onsubmit=async event=>{
  event.preventDefault();$('confirm-asset-import').disabled=true;$('asset-import-message').textContent='正在导入并校验文件…';
  try{
    const soma=await source('soma'),ply=await source('ply');
    const response=await fetch('/api/projects',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({soma,ply,initialize:$('asset-initialize').checked})});
    const result=await response.json();if(!response.ok)throw Error(result.error||'导入失败');
    location.href='/adam-forge-web-server/?project='+result.id;
  }catch(error){$('asset-import-message').textContent=error.message;}
  finally{$('confirm-asset-import').disabled=false;}
};

let browseKind='soma',parentDirectory='',directoryRequest=null;
async function browse(path=''){
  $('browser-select-directory').disabled=true;
  $('browser-message').textContent='正在读取目录…';$('browser-entries').textContent='';
  try{
    const query=new URLSearchParams({type:browseKind==='ply'?'ply':'npz'});if(path.trim())query.set('path',path.trim());
    const response=await fetch('/api/browse?'+query),result=await response.json();
    if(!response.ok)throw Error(result.error||'无法读取目录');
    $('browser-directory').value=result.path;parentDirectory=result.parent;
    $('browser-select-directory').disabled=false;
    $('browser-parent').disabled=result.parent===result.path;
    $('browser-message').textContent=result.truncated?'只显示前 500 项，请进入更具体的目录。':(result.entries.length?'选择目录进入，点击文件填入路径。':'当前目录没有匹配文件或子目录。');
    for(const entry of result.entries){
      if(browseKind==='directory'&&!entry.directory)continue;
      const button=document.createElement('button');button.type='button';button.className='browser-entry';
      const icon=document.createElement('span');icon.textContent=entry.directory?'▸':'◇';
      const name=document.createElement('span');name.textContent=entry.name;button.append(icon,name);
      button.onclick=()=>{if(entry.directory)browse(entry.path);else{$('asset-'+browseKind+'-path').value=entry.path;$('server-browser').close();}};
      $('browser-entries').append(button);
    }
  }catch(error){$('browser-message').textContent=error.message;}
}
for(const kind of ['soma','ply'])$('browse-'+kind).onclick=()=>{
  browseKind=kind;$('browser-select-directory').hidden=true;$('browser-title').textContent='浏览服务器 · '+(kind==='soma'?'SOMA .npz':'Gaussian .ply');
  $('server-browser').showModal();browse($('asset-'+kind+'-path').value);
};
$('close-server-browser').onclick=()=>$('server-browser').close();
$('browser-parent').onclick=()=>browse(parentDirectory);
$('browser-go').onclick=()=>browse($('browser-directory').value);
$('browser-directory').onkeydown=event=>{if(event.key==='Enter'){event.preventDefault();browse(event.target.value);}};

export function chooseServerDirectory(path=''){
  return new Promise((resolve,reject)=>{
    directoryRequest={resolve,reject};browseKind='directory';
    $('browser-title').textContent='浏览服务器 · 选择导出目录';
    $('browser-select-directory').hidden=false;$('browser-select-directory').disabled=true;
    $('server-browser').showModal();browse(path);
  });
}
$('browser-select-directory').onclick=()=>{
  if(!directoryRequest)return;
  const {resolve}=directoryRequest;directoryRequest=null;
  resolve($('browser-directory').value);$('server-browser').close();
};
$('server-browser').addEventListener('close',()=>{
  if(directoryRequest){directoryRequest.reject(new DOMException('已取消选择','AbortError'));directoryRequest=null;}
});
