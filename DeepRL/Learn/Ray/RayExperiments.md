# 설치 

pip install ray[rllib]

# 실험

## a2c 

Breakout-v4 를 대상으로 실행한다. 
yaml 파일을 만들고 실행한다. 
메모리를 많이 사용하여 2개 워커만 실행하고 learner는 GPU를 사용한다. 

PS D:\nolme\ggp\DeepRL\Learn\Ray> rllib train --ray-object-store-memory 500000000 -f .\breakout-a2c.yaml

위와 같이 옵션을 주고 object_store_memory를 줄이려고 해도 줄지가 않는다. 
ray의 분산 처리 구조는 메모리에 기초하고 메모리 사용량이 너무 크다. 


## r2d2 

구현되어 있다고 하나 실제 사용을 하려면 작업할 내용이 많아 보인다. 
메모리 사용이 매우 많은 이유를 알아본다. 

