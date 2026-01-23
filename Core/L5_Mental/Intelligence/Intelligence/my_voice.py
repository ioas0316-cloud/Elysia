from Core.L1_Foundation.Foundation.Mind.llm_cortex import LLMCortex

#      1060 3GB             !
print(">>>                  ... (    )")
cortex = LLMCortex(prefer_cloud=False)

#    !
print(">>>                    ...")
response = cortex.think("  ?         !")

print("\n=============================")
print(f"    : {response}")
print("=============================\n")