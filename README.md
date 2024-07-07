# DEBATE (ACL 2024 findings)
DEBATE: Devil’s Advocate-Based Assessment and Text Evaluation
Alex G. Kim* Keonwoo Kim* Sangwon Yoon* (*Equal Contribution)

https://arxiv.org/pdf/2405.09935

## Abstract
As natural language generation (NLG) models have become prevalent, systematically assessing the quality of machine-generated texts has become increasingly important. Recent studies introduce LLM-based evaluators that operate as reference-free metrics, demonstrating their capability to adeptly handle novel tasks. However, these models generally rely on a single-agent approach, which, we argue, introduces an inherent limit to their performance. This is because there exist biases in LLM agent’s responses, including preferences for certain text structure or content. In this work, we propose DEBATE, an NLG evaluation framework based on multiagent scoring system augmented with a concept of Devil’s Advocate. Within the framework, one agent is instructed to criticize other agents’ arguments, potentially resolving the bias in LLM agent’s answers. DEBATE substantially outperforms the previous state-of-the-art methods in two meta-evaluation benchmarks in NLG evaluation, SummEval and TopicalChat. We also show that the extensiveness of debates among agents and the persona of an agent can influence the performance of evaluators.

<p align="center">
<img src=".\png\figure" height = "350" alt="" align=center />
</p>

## Main Result
<p align="center">
<img src=".\png\results" height = "450" alt="" align=center />
</p>

## Citation
If you find this repo useful, please cite our paper. 

```
@misc{kim2024debatedevilsadvocatebasedassessment,
      title={DEBATE: Devil's Advocate-Based Assessment and Text Evaluation}, 
      author={Alex Kim and Keonwoo Kim and Sangwon Yoon},
      year={2024},
      eprint={2405.09935},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.09935}, 
}```
