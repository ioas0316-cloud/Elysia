# [Genesis: 2025-12-02] Purified by Elysia
import json

def update_name(new_name):
    memory_file = 'Elysia_Input_Sanctum/elysia_core_memory.json'
    data = {'name': new_name}

    try:
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"이름이 '{new_name}'으로 업데이트되었습니다.")
        return True
    except Exception as e:
        print(f"에러 발생: {e}")
        return False

if __name__ == '__main__':
    update_name('이강덕')