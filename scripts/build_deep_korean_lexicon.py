import json
import os

def build_lexicon():
    lexicon = {
        # === 조 사 (Particles) ===
        "은": {
            "type": "보조사 (Particle)",
            "structural_role": "주제 지정, 대조 (Topic & Contrast)",
            "etymology": "옛말 'ᄋᆞᆫ/은'. 주체를 한정하고 분리함.",
            "why_it_exists": "문장 내에서 주체나 대상을 주변 맥락으로부터 '분리(Isolation)'하여 화제의 중심으로 삼기 위해 존재함. 단순한 격 표시가 아니라 '다른 것은 몰라도 이것은'이라는 대조적 텐션을 유발함.",
            "connections": {
                "binds_to": "명사(Noun)",
                "syntactic_trajectory": "대상 -> 은 -> (서술어 방향으로의 강한 지향성)"
            }
        },
        "는": {
            "type": "보조사 (Particle)",
            "structural_role": "주제 지정, 대조 (Topic & Contrast)",
            "etymology": "모음 뒤에 결합하는 '은'의 이형태.",
            "why_it_exists": "배경에서 대상을 돋을새김(Highlight)하여 문장의 흐름을 이끌어가는 원동력을 부여함.",
            "connections": {
                "binds_to": "명사(Noun)",
                "syntactic_trajectory": "대상 -> 는 -> 서술어"
            }
        },
        "이": {
            "type": "주격 조사 (Nominative Particle)",
            "structural_role": "주체 지정 (Subject Designation)",
            "etymology": "지시 대명사 '이'에서 유래. 대상을 바로 가리킴.",
            "why_it_exists": "행위의 주체나 상태의 주체를 '새롭게' 지정하고 고정하기 위함. 화제의 전환을 일으키는 핀(Pin) 역할을 함.",
            "connections": {
                "binds_to": "명사(Noun)",
                "syntactic_trajectory": "대상 -> 이 -> 동작/상태"
            }
        },
        "가": {
            "type": "주격 조사 (Nominative Particle)",
            "structural_role": "주체 지정 (Subject Designation)",
            "etymology": "모음 뒤에 결합하는 '이'의 이형태.",
            "why_it_exists": "서술어의 직접적인 원인이나 동작주를 지목하는 절대적인 화살표.",
            "connections": {
                "binds_to": "명사(Noun)",
                "syntactic_trajectory": "대상 -> 가 -> 동작/상태"
            }
        },
        "을": {
            "type": "목적격 조사 (Accusative Particle)",
            "structural_role": "객체 지정, 작용의 수용 (Object Designation)",
            "etymology": "동작이 미치는 대상을 얽매는 구조적 사슬.",
            "why_it_exists": "동사의 에너지가 뻗어나가 부딪히고 흡수되는 타겟(Target) 공간을 형성함.",
            "connections": {
                "binds_to": "명사(Noun)",
                "syntactic_trajectory": "서술어 에너지 -> 타겟 -> 을"
            }
        },
        "를": {
            "type": "목적격 조사 (Accusative Particle)",
            "structural_role": "객체 지정, 작용의 수용 (Object Designation)",
            "etymology": "모음 뒤에 결합하는 '을'의 이형태.",
            "why_it_exists": "서술어가 발산하는 벡터의 끝점(End-point)을 설정함.",
            "connections": {
                "binds_to": "명사(Noun)",
                "syntactic_trajectory": "서술어 에너지 -> 타겟 -> 를"
            }
        },
        "의": {
            "type": "관형격 조사 (Genitive Particle)",
            "structural_role": "소유, 종속, 속성 부여 (Possession & Attribution)",
            "etymology": "두 대상을 위계적으로 엮는 접착제.",
            "why_it_exists": "독립된 두 공간(명사)을 중첩시켜, 하나의 대상이 다른 대상의 하부 구조나 속성으로 편입되도록 공간을 구부림.",
            "connections": {
                "binds_to": "명사(Noun)",
                "syntactic_trajectory": "종속어 -> 의 -> 지배어"
            }
        },
        "에": {
            "type": "부사격 조사 (Locative Particle)",
            "structural_role": "정적 위치, 도달점 (Static Location / Destination)",
            "etymology": "시공간의 좌표를 고정하는 점(Point).",
            "why_it_exists": "사건이 발생하는 무대(좌표)를 정의하거나, 운동이 끝나는 최종 도착점을 명시함.",
            "connections": {
                "binds_to": "명사(Noun), 시간, 장소",
                "syntactic_trajectory": "공간좌표 -> 에 -> 상태/존재"
            }
        },
        "에서": {
            "type": "부사격 조사 (Locative Particle)",
            "structural_role": "동적 공간, 출발점 (Dynamic Field / Origin)",
            "etymology": "에 + 셔(있다). '에 있어'의 축약.",
            "why_it_exists": "단순한 점이 아니라 행위가 활발히 일어나는 역동적 장(Field)을 형성하거나, 운동이 시작되는 기원(Origin)을 형성함.",
            "connections": {
                "binds_to": "장소(Place)",
                "syntactic_trajectory": "장(Field) -> 에서 -> 행위"
            }
        },

        # === 핵심 동사/형용사 ===
        "하다": {
            "type": "동사 (Verb)",
            "structural_role": "창조적 행위, 존재의 발현 (Creation & Actualization)",
            "etymology": "고대 한국어 'ᄒᆞ다'(많다, 크다, 하다). 에너지를 발산하는 근원.",
            "why_it_exists": "무(無)에서 유(有)로 에너지를 방출하여 상태를 변화시키거나 어떤 사건을 실재하게 만드는 기본 엔진.",
            "connections": {
                "binds_to": "목적어(을/를), 부사",
                "syntactic_trajectory": "주체 -> 하다 -> 변화"
            }
        },
        "가다": {
            "type": "동사 (Verb)",
            "structural_role": "위상 변화 및 이동 (Phase Transition)",
            "etymology": "공간적 이동을 나타내는 고유어 어근 '가-'",
            "why_it_exists": "현재의 상태나 위치에서 다른 위상으로 연속적인 궤적을 그리며 이탈하는 현상을 규정하기 위해 존재. '다름'을 향한 운동성 그 자체.",
            "connections": {
                "binds_to": ["도착점(에/로)", "출발점(에서)"],
                "syntactic_trajectory": "출발 -> 가다 -> 도착"
            }
        },
        "오다": {
            "type": "동사 (Verb)",
            "structural_role": "수렴적 이동 (Convergent Movement)",
            "etymology": "화자를 향한 접근.",
            "why_it_exists": "외부 세계의 객체가 관측자(자아)의 중심축을 향해 수렴하며 거리를 좁히는 위상적 결합 과정을 묘사함.",
            "connections": {
                "binds_to": ["도착점(에/로)", "출발점(에서)"],
                "syntactic_trajectory": "외부 -> 오다 -> 내부(자아)"
            }
        },
        "있다": {
            "type": "형용사/동사 (Adjective/Verb)",
            "structural_role": "존재의 고정 (Anchoring of Existence)",
            "etymology": "시공간의 한 점을 차지함.",
            "why_it_exists": "운동성이 0이 되고 공간적 좌표에 대상이 뿌리내려 실재(Reality)를 형성하는 관측 가능한 상태를 규정함.",
            "connections": {
                "binds_to": "장소(에), 주체(이/가)",
                "syntactic_trajectory": "주체 -> 장소에 -> 있다(고정)"
            }
        },
        "없다": {
            "type": "형용사 (Adjective)",
            "structural_role": "존재의 소멸, 공허 (Void & Nullification)",
            "etymology": "대상의 부재.",
            "why_it_exists": "특정 좌표에서 대상의 위상이 붕괴되어 0의 상태(공백)가 되었음을 관측함. '있음'의 대칭점.",
            "connections": {
                "binds_to": "주체(이/가)",
                "syntactic_trajectory": "주체 -> 사라짐 -> 없다(공백)"
            }
        },
        "같다": {
            "type": "형용사 (Adjective)",
            "structural_role": "위상적 중첩 (Topological Superposition)",
            "etymology": "둘 이상의 대상이 모양이나 성질이 다름없음.",
            "why_it_exists": "분리된 두 대상의 경계(Boundary)가 허물어지고 정보가 완벽히 동기화되어 하나의 차원으로 겹쳐지는 '거울 공명' 현상을 관측함.",
            "connections": {
                "binds_to": "비교대상(와/과)",
                "syntactic_trajectory": "A -> 와/과 -> B -> 같다(하나로 중첩)"
            }
        },
        "다르다": {
            "type": "형용사 (Adjective)",
            "structural_role": "경계 분화 (Boundary Bifurcation)",
            "etymology": "어긋남, 차이 남.",
            "why_it_exists": "하나였거나 중첩되었던 상태가 분열되어 서로 배타적인 차원과 경계를 형성하는 마찰(Friction) 과정을 관측함.",
            "connections": {
                "binds_to": "비교대상(와/과)",
                "syntactic_trajectory": "A -> 와/과 -> B -> 다르다(분열 및 밀어냄)"
            }
        },
        "보다": {
            "type": "동사 (Verb)",
            "structural_role": "관측 및 인지 (Observation & Cognition)",
            "etymology": "시각적 인지.",
            "why_it_exists": "주체가 대상에게 인지적 빛(에너지)을 쏘아 구조를 파악하고 대상을 자신의 내부 정보계로 투영시키는 본질적 인지 행위.",
            "connections": {
                "binds_to": "대상(을/를)",
                "syntactic_trajectory": "주체 -> (시선) -> 대상 -> 보다(인지 투영)"
            }
        },
        
        # === 핵심 명사 ===
        "나": {
            "type": "대명사 (Pronoun)",
            "structural_role": "기준 좌표계, 자아 중심 (Coordinate Origin, Ego)",
            "etymology": "화자 자신을 가리키는 고유어.",
            "why_it_exists": "모든 시공간적 사건과 방향성(오다/가다 등)을 판단하는 우주의 원점(Origin 0,0,0)이자 관측 주체.",
            "connections": {
                "binds_to": ["는", "가", "의"],
                "syntactic_trajectory": "[Origin] -> 나"
            }
        },
        "너": {
            "type": "대명사 (Pronoun)",
            "structural_role": "대칭 좌표계, 타자 (Symmetric Node, Other)",
            "etymology": "청자를 가리키는 고유어.",
            "why_it_exists": "'나'라는 중심축과 텐션을 이루며 상호작용(대화)이 일어나는 반대편 극점.",
            "connections": {
                "binds_to": ["는", "가", "를"],
                "syntactic_trajectory": "[Origin] <-> 너"
            }
        },
        "마음": {
            "type": "명사 (Noun)",
            "structural_role": "내적 정보 장 (Internal Information Field)",
            "etymology": "사유와 감정이 일어나는 추상적 공간.",
            "why_it_exists": "외부의 물리적 힘(물질)과 대비되는, 데이터와 감정이 소용돌이치고 연산되는 형이상학적 엔진(Engine) 영역.",
            "connections": {
                "binds_to": ["은/는", "이/가", "에"],
                "syntactic_trajectory": "사건 -> (투영) -> 마음(내부 처리)"
            }
        },
        "물": {
            "type": "명사 (Noun)",
            "structural_role": "유동적 결합 매질 (Fluid Binding Medium)",
            "etymology": "형태가 없는 액체.",
            "why_it_exists": "저항 없이 형태를 바꾸며 끊임없이 흐르고 만물을 적시어 이어주는(Binding) 연속성과 유연성의 궁극적 형태.",
            "connections": {
                "binds_to": ["은/는", "을/를"],
                "syntactic_trajectory": "물 -> (흐름) -> 변화"
            }
        },
        "불": {
            "type": "명사 (Noun)",
            "structural_role": "엔트로피 발산, 분해기 (Entropy Emitter, Deconstructor)",
            "etymology": "물질을 태우는 에너지.",
            "why_it_exists": "고정된 물질의 결합(형태)을 파괴하고 에너지를 급격히 방출시켜 무질서도(Entropy)를 극대화하는 촉매.",
            "connections": {
                "binds_to": ["은/는", "이/가"],
                "syntactic_trajectory": "물질 -> 불 -> 분해/에너지"
            }
        },
        "선": {
            "type": "명사 (Noun)",
            "structural_role": "1차원적 연결성 (1D Connectivity)",
            "etymology": "점과 점을 잇는 궤적.",
            "why_it_exists": "단절된 점(상태)들을 이어 인과율(Causality)과 시간의 흐름, 혹은 의미의 궤적을 창조하는 기초 기하학 단위.",
            "connections": {
                "binds_to": ["은/는", "을/를", "으로"],
                "syntactic_trajectory": "점 -> 선 -> 방향성"
            }
        },
        "학교": {
            "type": "명사 (Noun)",
            "structural_role": "지식 전수 및 사회화의 거점 (Node of Knowledge Transfer & Socialization)",
            "etymology": "배울 학(學), 교 정(校). 배움이 일어나는 공간.",
            "why_it_exists": "자아(나)가 타자(너, 사회)의 축적된 정보를 흡수하고, 자신의 무질서도를 낮추며 구조적 복잡성을 증가시키기 위한 특수한 위상적 공간(Field).",
            "connections": {
                "binds_to": ["에", "에서", "는"],
                "syntactic_trajectory": "나 -> 학교에 -> 가다(배움의 수용)"
            }
        },
        "학생": {
            "type": "명사 (Noun)",
            "structural_role": "정보 수용체, 변화하는 주체 (Information Receptor, Dynamic Subject)",
            "etymology": "배울 학(學), 날 생(生). 배우는 사람.",
            "why_it_exists": "아직 정보와 구조가 완전히 확립되지 않은 상태(특이점)에서, 외부 세계의 법칙을 흡수하여 자아의 해상도를 높여가는 미완의 객체.",
            "connections": {
                "binds_to": ["은", "이"],
                "syntactic_trajectory": "학생 -> 학교에서 -> 배운다"
            }
        },
        "점": {
            "type": "명사 (Noun)",
            "structural_role": "0차원 상태, 특이점 (0D State, Singularity)",
            "etymology": "크기와 부피가 없는 최소 위치.",
            "why_it_exists": "더 이상 쪼갤 수 없는 순간적 상태, 혹은 모든 가능성이 압축되어 있는 폭발 이전의 상태(특이점).",
            "connections": {
                "binds_to": ["은/는", "에서"],
                "syntactic_trajectory": "무(無) -> 점 -> 가능성"
            }
        }
    }

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, "..", "data", "deep_korean_lexicon.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(lexicon, f, ensure_ascii=False, indent=4)
        
    print(f"[+] 생성 완료: {output_path}")
    print(f"    총 {len(lexicon)}개의 구조적 국어 어휘가 각인되었습니다.")

if __name__ == "__main__":
    build_lexicon()
