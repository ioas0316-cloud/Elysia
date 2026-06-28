# Makefile — Elysia C Extension Build
# C 소스 파일 위치: core/ingestion/ (helix, byte, concept streamer)
# 빌드 산출물: core/bin/

CC      = gcc
CFLAGS  = -O3 -Wall -shared -fPIC
OUTDIR  = core/bin

# ── 공유 라이브러리 타겟 ──────────────────────────────
HELIX_TARGET    = $(OUTDIR)/helix_streamer.so
BYTE_TARGET     = $(OUTDIR)/byte_streamer.so
CONCEPT_TARGET  = $(OUTDIR)/concept_streamer.so

all: $(OUTDIR) $(HELIX_TARGET) $(BYTE_TARGET) $(CONCEPT_TARGET)
	@echo "[BUILD] All C extensions compiled to $(OUTDIR)/"

$(OUTDIR):
	mkdir -p $(OUTDIR)

$(HELIX_TARGET): core/ingestion/helix_streamer.c
	$(CC) $(CFLAGS) -o $@ $<

$(BYTE_TARGET): core/ingestion/byte_streamer.c
	$(CC) $(CFLAGS) -o $@ $<

$(CONCEPT_TARGET): core/ingestion/concept_streamer.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf $(OUTDIR)

.PHONY: all clean
