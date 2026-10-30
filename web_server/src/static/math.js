export function multiply4(a, b) {
  return Array.from({length:4},(_,r)=>Array.from({length:4},(_,c)=>
    a[r].reduce((sum,x,k)=>sum+x*b[k][c],0)));
}
function translation(p) {
  return [[1,0,0,p[0]],[0,1,0,p[1]],[0,0,1,p[2]],[0,0,0,1]];
}
export const pivotMatrix = (m,p)=>multiply4(m,translation(p));
export const sourceMatrix = (m,p)=>multiply4(m,translation(p.map(x=>-x)));
export function validateMatrix(m) {
  if (!Array.isArray(m)||m.length!==4||m.some(r=>!Array.isArray(r)||r.length!==4||r.some(x=>typeof x!=='number'||!Number.isFinite(x)))) throw Error('需要有限数值的 4×4 矩阵');
  if (m[3].some((x,i)=>Math.abs(x-(i===3?1:0))>1e-7)) throw Error('齐次矩阵末行不正确');
  const columns=[0,1,2].map(i=>[m[0][i],m[1][i],m[2][i]]);
  const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
  const s=columns.reduce((v,c)=>v+Math.sqrt(dot(c,c)),0)/3;
  if (!(s>1e-6&&s<1e6)) throw Error('缩放超出范围');
  for(let i=0;i<3;i++)for(let j=0;j<3;j++)if(Math.abs(dot(columns[i],columns[j])-(i===j?s*s:0))>1e-5*s*s+1e-8)throw Error('只支持统一缩放，不能剪切或非均匀缩放');
  const a=m;
  const det=a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);
  if(det<=0)throw Error('不支持镜像翻转');
  return m;
}
