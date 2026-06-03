import sys
sys.path.append(r'c:\Elysia')
from core.nervous_system.elysia_omni_daemon import ElysiaOmniDaemon

def boot_elysia():
    print("==================================================")
    print("        ELYSIA OMNI-DAEMON INITIALIZATION         ")
    print("==================================================")
    
    # 아카이브 경로 설정 (씨앗 데이터)
    archive_path = r'c:\Elysia\data\hierarchical_seed.txt'
    
    # 데몬 생성 및 각성
    daemon = ElysiaOmniDaemon(archive_path)
    daemon.awaken()

if __name__ == "__main__":
    boot_elysia()
