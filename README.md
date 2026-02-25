First install stride
then in the same conda environment
```
git clone
cd diffusion-fwi-2d
```

optional but recommended: 
```conda uninstall pytorch```

```
pip install -r requirements.txt
```

if running into memory errors you can do
```
mkdir ~/tmp
TMPDIR=~/tmp pip install -r requirements txt
```

then finally

```pip install -e .```