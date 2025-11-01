import chat_interface as chat

name = '민수'
print('Setting name to:', name)
ok = chat.set_identity(name)
print('Saved:', ok)
cm = chat.load_core_memory()
print('Core memory identity:', cm.get('identity'))
