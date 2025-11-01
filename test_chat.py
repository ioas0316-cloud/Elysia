from chat_interface import ChatInterface

def main():
    chat = ChatInterface()
    
    # 테스트 시나리오
    test_inputs = [
        ("안녕하세요!", {'type': 'greeting', 'speaker': '이강덕'}),
        ("오늘 날씨가 참 좋네요.", {'type': 'casual', 'speaker': '이강덕'}),
        ("AI에 대해 어떻게 생각하세요?", {'type': 'philosophical', 'speaker': '이강덕'})
    ]
    
    for user_input, context in test_inputs:
        print(f"\n사용자: {user_input}")
        response = chat.process_input(user_input, context)
        print(f"Elysia: {response}")

if __name__ == "__main__":
    main()