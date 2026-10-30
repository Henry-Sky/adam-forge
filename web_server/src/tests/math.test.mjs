import test from 'node:test';
import assert from 'node:assert/strict';
import {multiply4,pivotMatrix,sourceMatrix,validateMatrix} from '../static/math.js';

test('pivot manipulation exports the same original-camera transformation',()=>{
  const m=[[0,-.5,0,.2],[.5,0,0,-.1],[0,0,.5,1],[0,0,0,1]],p=[.3,.2,2.1];
  const back=sourceMatrix(pivotMatrix(m,p),p);
  back.flat().forEach((x,i)=>assert.ok(Math.abs(x-m.flat()[i])<1e-12));
  validateMatrix(back);
});
test('manual world translation composes in the exported coordinate frame',()=>{
  const m=[[.5,0,0,0],[0,-.5,0,0],[0,0,-.5,1],[0,0,0,1]],p=[0,0,2];
  const moved=multiply4([[1,0,0,.1],[0,1,0,.2],[0,0,1,.3],[0,0,0,1]],pivotMatrix(m,p));
  const out=sourceMatrix(moved,p);
  assert.ok(Math.abs(out[2][3]-1.3)<1e-12);
  assert.equal(out[0][3],.1);assert.equal(out[1][3],.2);
});
test('reject non-uniform scale, reflection and malformed matrices',()=>{
  for(const m of [null,[[1]],[[1,0,0,0],[0,2,0,0],[0,0,1,0],[0,0,0,1]],[[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]],[[NaN,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]])assert.throws(()=>validateMatrix(m));
});
