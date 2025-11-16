import argparse
import os
import sys
import time
from pathlib import Path

import pygame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=800)
    ap.add_argument('--seconds', type=int, default=4)
    args = ap.parse_args()

    log_path = Path('logs') / 'diag_pygame_env.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        except Exception:
            pass

    drv = os.environ.get('SDL_VIDEODRIVER', '')
    aud = os.environ.get('SDL_AUDIODRIVER', '')
    log(f"SDL_VIDEODRIVER={drv} SDL_AUDIODRIVER={aud}")

    try:
        pygame.init()
        screen = pygame.display.set_mode((args.size, args.size))
        pygame.display.set_caption(f"Diag: {drv or 'default'} (ESC to exit)")
    except Exception as ex:
        log(f"display init failed: {ex}")
        print(f"[실패] display init: {ex}")
        return 1

    font = pygame.font.SysFont(None, 24)
    start = time.time()
    ok = False
    while True:
        for e in pygame.event.get():
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                ok = True
                pygame.quit()
                print("[사용자 종료] ESC")
                return 0
        screen.fill((20, 30, 40))
        msg = f"Driver: {drv or 'default'} | Close in {max(0, int(args.seconds-(time.time()-start)))}s"
        surf = font.render(msg, True, (235,235,245))
        screen.blit(surf, (10, 10))
        pygame.display.flip()
        if time.time() - start > args.seconds:
            ok = True
            break
        pygame.time.delay(16)

    pygame.quit()
    print("[성공] 창 생성/유지 OK")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

