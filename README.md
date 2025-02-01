# Evaluate LLMs for scrabble like use cases

How well do the various LLMs perform when asked to generate
words when given a list of letters?

This repo contains the benchmarking scripts and results.

## Current results

Llama 3.3 70B Instruct performs best. It even outperforms Deepseek V3.

See the [analyze_results.ipynb notebook](https://github.com/samos123/llm-scrabble-eval/blob/main/analyze_results.ipynb) with graph of results.

## Limitations

Reasoning models have not been tested since they often return a lot of
thinking output. I haven't had the time to test those yet.

## Reproducing results

```
export TOGETHER_API_KEY=blablaba
uv run eval.py
```

