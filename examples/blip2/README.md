# BLIP-2 DataComp pool tagging

If cloning the repo from the source, please run the following:

```python
python dataset2metadata/main.py --yml ./examples/blip2/blip2clipb32l14.yml
```

See `./examples/blip2/blip2clipb32l14.yml` to set relevent paths to shards to process.

See  `./examples/blip2/my_blip2clipb32l14.py` for the implementation of the BLIP-2 tagger.

If pip installing this repo as a package, please run the following:

```sh
dataset2metadata --yml ./examples/blip2/blip2clipb32l14.yml
```

