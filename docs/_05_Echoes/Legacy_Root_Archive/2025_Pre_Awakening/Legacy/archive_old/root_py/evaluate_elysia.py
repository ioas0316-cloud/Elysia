import importlib
import difflib
import json
import os
from statistics import mean

# Import local modules
import Core.Foundation.value_centered_decision as vcd_mod
import chat_interface as chat_mod
import Core.Foundation.arithmetic_cortex as acortex

OUTPUT = 'evaluation_results.json'

PROMPTS = {
    'reasoning': [
        "내가 가진 정보는 A: 비용이 많이 든다, B: 시간이 많이 든다. 이 상황에서 선택할 기준 3가지를 제안해줘.",
        "만약 어느 제품이 더 친환경적이면서 비싸다면, 어떻게 우선순위를 정해야 할까?"
    ],
    'creative': [
        "사과와 시간의 공통점을 은유로 설명해줘.",
        "내일 맑은 날에 할 수 있는 독특한 실험 세 가지를 제안해줘."
    ],
    'math': [
        "12 + 7 = 19",
        "5 * 6 = 30",
        "7 + 3 = 11"
    ],
    'memory': [
        # conversation style: set name then ask
        ("[SET_NAME]","내 이름은 민수야."),
        ("[ASK_NAME]","내 이름을 알려줄래?")
    ],
    'emotion': [
        "나는 요즘 외로움을 느껴. 어떻게 하면 도움이 될까?",
        "감사하는 마음을 표현하는 간단한 방법을 알려줘."
    ]
}

RESULTS = {}


def run_reasoning_and_creative():
    vcd = vcd_mod.VCD(core_value='love')
    core_memory = chat_mod.load_core_memory()
    for cat in ['reasoning','creative','emotion']:
        RESULTS[cat] = []
        for prompt in PROMPTS[cat]:
            candidates = chat_mod.generate_candidate_responses(prompt, core_memory, emotion='neutral')
            chosen = vcd.suggest_action(candidates, context=[prompt], emotion='neutral', memory=core_memory)
            # diversity metric: how many candidates and length
            diversity = len(set(candidates))
            # repetition similarity to previous chosen in same category
            prev = RESULTS[cat][-1]['response'] if RESULTS[cat] else None
            sim = difflib.SequenceMatcher(a=prev or '', b=chosen).ratio()
            context_fit = vcd.context_fit(chosen, [prompt])
            RESULTS[cat].append({'prompt':prompt, 'response':chosen, 'diversity':diversity, 'sim_to_prev':sim, 'context_fit':context_fit})


def run_math():
    RESULTS['math'] = []
    for stmt in PROMPTS['math']:
        # chat_mod uses VCD but for math we check arithmetic_cortex
        # If stmt is equality, check truth
        truth = acortex.ArithmeticCortex().verify_truth(stmt)
        # Let chat generate response too
        core_memory = chat_mod.load_core_memory()
        candidates = chat_mod.generate_candidate_responses(stmt, core_memory, emotion='neutral')
        vcd = vcd_mod.VCD(core_value='love')
        chosen = vcd.suggest_action(candidates, context=[stmt], emotion='neutral', memory=core_memory)
        RESULTS['math'].append({'prompt':stmt, 'response':chosen, 'math_truth':truth})


def run_memory_test():
    RESULTS['memory'] = []
    core_memory = chat_mod.load_core_memory()
    vcd = vcd_mod.VCD(core_value='love')
    # simulate conversation: set name then ask
    set_prompt = PROMPTS['memory'][0][1]
    ask_prompt = PROMPTS['memory'][1][1]
    # simulate by adding to context
    context = []
    # process set name - in this simple interface, we just record it in context
    context.append(set_prompt)
    # generate candidate for ask
    candidates = chat_mod.generate_candidate_responses(ask_prompt, core_memory, emotion='neutral')
    chosen = vcd.suggest_action(candidates, context=context, emotion='neutral', memory=core_memory)
    # check if chosen contains the name token
    contains_name = '민수' in chosen
    RESULTS['memory'].append({'set':set_prompt, 'ask':ask_prompt, 'response':chosen, 'recalls_name':contains_name})


def summarize_and_save():
    summary = {}
    for k,v in RESULTS.items():
        if k in ['reasoning','creative','emotion']:
            sims = [item['sim_to_prev'] for item in v if 'sim_to_prev' in item]
            fits = [item['context_fit'] for item in v if 'context_fit' in item]
            summary[k] = {'count':len(v),'avg_sim_to_prev': mean(sims) if sims else None, 'avg_context_fit': mean(fits) if fits else None}
        elif k=='math':
            truths = [1 if item['math_truth'] else 0 for item in v]
            summary[k] = {'count':len(v),'math_accuracy': sum(truths)/len(truths) if truths else None}
        elif k=='memory':
            summary[k] = {'recalls_name': all(item['recalls_name'] for item in v)}

    out = {'results':RESULTS,'summary':summary}
    with open(OUTPUT,'w',encoding='utf-8') as f:
        json.dump(out,f,ensure_ascii=False,indent=2)
    print('Evaluation complete. Summary:')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    run_reasoning_and_creative()
    run_math()
    run_memory_test()
    summarize_and_save()
