import os, json

path = os.path.join('/home/qua/Data/reaction_data/output_conversation_rewritten.jsonl')

seeks = [0]
with open(path, 'rb') as f:
    while True:
        line = f.readline()
        if not line:
            break
        seeks.append(f.tell())
seeks = seeks[:-1]  # 去掉EOF位置

with open(path, 'a') as f:
    f.write(json.dumps(seeks))
