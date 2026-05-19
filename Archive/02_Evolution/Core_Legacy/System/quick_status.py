from Core.System.web_knowledge_connector import WebKnowledgeConnector

w = WebKnowledgeConnector()
m = w.comm_enhancer.get_communication_metrics()
print(f'     : {m["vocabulary_size"]} ')
print(f'  : {m["expression_patterns"]}')
print(f'   : {m["dialogue_templates"]}')
