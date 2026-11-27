"""Corpus preparation: tokenize all of TinyStoriesV2-GPT4-train and split it into TRAIN/DEV/TEST
deterministically (SHA-256 of the normalized text, modulo 10,000).
Variables: STERA_CORPUS (.txt file), STERA_TOKENIZER (tokenizer.json), STERA_DATA (output folder).
"""
import array,hashlib,json,mmap,os,struct,sys,time,zlib
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM']='false'
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parent
DATA=Path(os.environ.get('STERA_DATA') or os.environ['STERA_DATI'])
SOURCE=Path(os.environ['STERA_CORPUS'])
TOK=Path(os.environ['STERA_TOKENIZER'])
POOL=ROOT/'data/racconti_riservati_dev_2000.jsonl'
EXPECTED_SOURCE='6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def group(raw):return hashlib.sha256(b' '.join(raw.split()).lower()).digest()
def main():
 assert sys.byteorder=='little'
 DATA.mkdir(parents=True,exist_ok=True)
 assert not (DATA/'MANIFEST.json').exists(),'Already prepared; preserve corpus'
 assert sha(SOURCE)==EXPECTED_SOURCE
 assert sha(TOK)=='f6ed3d307010c244c22aeffbde05f419cf277c23e64cf98b673cac5449cfeff5'
 tok=Tokenizer.from_file(str(TOK));assert tok.get_vocab_size()==50257
 reserved=set()
 with SOURCE.open('rb') as f:
  mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
  for line in POOL.read_text().splitlines():
   off=json.loads(line)['source_story_byte_offset'];end=mm.find(b'<|endoftext|>',off)
   assert end>off;reserved.add(group(mm[off:end].strip()))
  mm.close()
 counts=[0,0,0];targets=[0,0,0];seen=0;duplicates=0;hashes=set();start=time.monotonic()
 pack=DATA/'stories.blocks.partial';index=DATA/'stories.index.partial'
 block=array.array('H');pending=[];block_count=0
 with pack.open('xb') as fp,index.open('xb') as fi:
  def flush():
   nonlocal block,pending,block_count
   if not pending:return
   raw=block.tobytes();compressed=zlib.compress(raw,3);offset=fp.tell()
   fp.write(struct.pack('<II',len(compressed),len(raw)));fp.write(compressed)
   for tokenoff,n,sourceoff,split in pending:fi.write(struct.pack('<QIIII',offset,tokenoff,n,sourceoff,split))
   block=array.array('H');pending=[];block_count+=1
  with SOURCE.open('rb') as fs:
   buffer=[];off=0;storyoff=0
   def accept(raw,sourceoff):
    nonlocal seen,duplicates
    raw=raw.strip()
    if not raw:return
    key=group(raw);bucket=int.from_bytes(key[:8],'little')%10000
    # 0=train,1=validation,2=test. Reserved historical evaluation pool stays VAL.
    split=1 if key in reserved else (2 if bucket<100 else 1 if bucket<200 else 0)
    text=raw.decode('utf-8',errors='strict')
    ids=[50256]+tok.encode(text,add_special_tokens=False).ids+[50256]
    assert max(ids)<50257 and len(ids)>2
    tokenoff=len(block);block.extend(ids);pending.append((tokenoff,len(ids),sourceoff,split))
    counts[split]+=1;targets[split]+=len(ids)-1;seen+=1
    if key in hashes:duplicates+=1
    else:hashes.add(key)
    if len(pending)>=256:flush()
    if seen%10000==0:
     fp.flush();fi.flush()
     free=__import__('shutil').disk_usage(DATA).free
     status=dict(stories=seen,split_stories=counts,split_targets=targets,blocks=block_count,
                 compressed_bytes=fp.tell(),index_bytes=fi.tell(),free_gib=free/2**30,elapsed_s=round(time.monotonic()-start,1))
     (DATA/'PROGRESS.json').write_text(json.dumps(status,indent=2));print('PREP_FULL',json.dumps(status),flush=True)
     assert free>=int(1.45*2**30),'Need disk reserve for native full-state checkpoints; partial corpus preserved'
   for line in fs:
    if line.strip()==b'<|endoftext|>':
     accept(b''.join(buffer),storyoff);buffer=[];storyoff=off+len(line)
    else:buffer.append(line)
    off+=len(line)
   accept(b''.join(buffer),storyoff)
  flush();fp.flush();fi.flush();os.fsync(fp.fileno());os.fsync(fi.fileno())
 assert seen>2000000 and all(n>10000 for n in counts)
 finalpack=DATA/'stories.blocks';finalindex=DATA/'stories.index'
 assert not finalpack.exists() and not finalindex.exists()
 pack.rename(finalpack);index.rename(finalindex)
 manifest=dict(schema='STERA_TS68_ALL_GPT4_TRAIN_CONTENT_SPLIT_V1',source=str(SOURCE),source_sha256=EXPECTED_SOURCE,
   tokenizer=str(TOK),tokenizer_sha256=sha(TOK),vocab=50257,eos=50256,bos=50256,seed=20261117,
   pack_sha256=sha(finalpack),index_sha256=sha(finalindex),stories=seen,blocks=block_count,
   split_stories=counts,split_target_tokens=targets,split_names=['TRAIN','VALIDATION','TEST'],
   split_policy='content normalized whitespace/lowercase SHA256 modulo10000: <100 TEST,<200 VAL,else TRAIN; reserved2000 content groups forced VAL',
   duplicate_occurrences=duplicates,duplicate_groups_always_same_split=True,reserved_content_groups=len(reserved),
   full_source_used=True,no_narrative_quality_selection=True,no_story_length_truncation=True,
   index_format='24 bytes little-endian QIIII: block byte offset,token offset,token length,source byte offset,split',
   block_format='uint32 compressed bytes,uint32 raw bytes,zlib uint16 token sequence',
   test_used_for_gradient=False,official_sealed_test_opened=False,elapsed_s=time.monotonic()-start)
 with (DATA/'MANIFEST.json').open('x') as f:json.dump(manifest,f,indent=2)
 print('FULL_CORPUS_READY',json.dumps(manifest),flush=True)
if __name__=='__main__':main()
