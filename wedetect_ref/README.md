# WeDetect-Ref

## 📏 Evaluating on Refcoco

- Download our processed annotations and proposals from [here](https://huggingface.co/datasets/fushh7/eval_refcoco).
- Organize the data as follows:
```
|--wedetect_ref
  |--eval_grounding
    |--eval_refcoco
    |--eval.py
  |--sft.py
  |--sft_referring.py
```
- Modify the dataset path of `img_path` in the line 46 of `eval_grounding/eval.py`
- Then, evaluate on benchmarks:

```
cd eval_grounding
export PYTHONPATH=../

torchrun --nproc-per-node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=29500 eval.py --checkpoint /PATH/TO/CHECKPOINT --dataset refcoco --num_select 20
```

## 📍 Training & Finetuning

- Please see the files in `scripts` for the commands.
- All bounding boxes are in the `x1y1x2y2` format.
- The data format for Stage1 & Stage2 are:

```
{
    "id": "000321973",
    "image": "000321973.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nRender a clear and concise summary of the photo."
        },
        {
            "from": "gpt",
            "value": "a man playing a conga drum with kids around him"
        }
    ]
},
{
    "id": "refcoco_train_37616",
    "image": "COCO_train2014_000000064350.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nState a short description of the region <object>."
        },
        {
            "from": "gpt",
            "value": "far right guy"
        }
    ],
    "bounding_boxes": [
        [
            357.3,
            70.37,
            500.0,
            308.13
        ]
    ]
},
```
- In Stage3, users shoulf first extract 100 proposes per image and saving the results in a json file. 

```
{
  "file1": proposal list1, #[[x1y1x2y2], [x1y1x2y2], ...]
  "file2": proposal list2,
}
```

- The data format for Stage3 is:

```
[
  {
      "image": "2386621.jpg",
      "class_name": "The full bowl next to the white and full plate and of the cream colored dish.",
      "bounding_boxes": [
          [
              178,
              184,
              293,
              283
          ]
      ]
  },
]
```
