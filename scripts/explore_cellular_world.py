import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from Project_Sophia.core.world import World
from tools.kg_manager import KGManager

# Define core concepts that should have higher initial energy
CORE_CONCEPTS = [
    "value:love", "concept:빛", "concept:물", "concept:생명", "concept:시간", "concept:공간",
    "obsidian_note:사랑", "obsidian_note:생명", "obsidian_note:시간", "obsidian_note:공간",
    "obsidian_note:땅", "obsidian_note:하늘", "obsidian_note:태양", "obsidian_note:물", "obsidian_note:빛",
    "obsidian_note:에너지", "obsidian_note:세포", "obsidian_note:신", "obsidian_note:인간", "obsidian_note:언어",
    "obsidian_note:역사", "obsidian_note:우주", "obsidian_note:음악", "obsidian_note:의식주", "obsidian_note:이름",
    "obsidian_note:인사", "obsidian_note:일", "obsidian_note:입", "obsidian_note:자동차", "obsidian_note:자전거",
    "obsidian_note:잠", "obsidian_note:종아리", "obsidian_note:종이", "obsidian_note:지구", "obsidian_note:지하철",
    "obsidian_note:집", "obsidian_note:춤", "obsidian_note:코", "obsidian_note:태양", "obsidian_note:팔",
    "obsidian_note:하늘", "obsidian_note:허기", "obsidian_note:허리", "obsidian_note:허벅지", "obsidian_note:혀",
    "obsidian_note:가방", "obsidian_note:가족", "obsidian_note:감각", "obsidian_note:감정", "obsidian_note:강",
    "obsidian_note:계절", "obsidian_note:골반", "obsidian_note:공간", "obsidian_note:귀", "obsidian_note:길",
    "obsidian_note:꿈", "obsidian_note:낮", "obsidian_note:노래", "obsidian_note:놀이", "obsidian_note:눈",
    "obsidian_note:단위", "obsidian_note:달", "obsidian_note:동물", "obsidian_note:등", "obsidian_note:땅",
    "obsidian_note:말", "obsidian_note:목", "obsidian_note:무제", "obsidian_note:바다", "obsidian_note:바람",
    "obsidian_note:바퀴", "obsidian_note:발", "obsidian_note:밤", "obsidian_note:밥", "obsidian_note:배",
    "obsidian_note:버스", "obsidian_note:비행기", "obsidian_note:빛", "obsidian_note:빵", "obsidian_note:사랑",
    "obsidian_note:산", "obsidian_note:생명", "obsidian_note:성경", "obsidian_note:세포", "obsidian_note:손",
    "obsidian_note:시간", "obsidian_note:식물", "obsidian_note:신", "obsidian_note:신발", "obsidian_note:어깨",
    "obsidian_note:어둠", "obsidian_note:언어", "obsidian_note:얼굴", "obsidian_note:엉덩이", "obsidian_note:에너지",
    "obsidian_note:역사", "obsidian_note:우주", "obsidian_note:음악", "obsidian_note:의식주", "obsidian_note:이름",
    "obsidian_note:인사", "obsidian_note:일", "obsidian_note:입", "obsidian_note:자동차", "obsidian_note:자전거",
    "obsidian_note:잠", "obsidian_note:종아리", "obsidian_note:종이", "obsidian_note:지구", "obsidian_note:지하철",
    "obsidian_note:집", "obsidian_note:춤", "obsidian_note:코", "obsidian_note:태양", "obsidian_note:팔",
    "obsidian_note:하늘", "obsidian_note:허기", "obsidian_note:허리", "obsidian_note:허벅지", "obsidian_note:혀",
    "obsidian_note:밀리그램", "obsidian_note:미터", "obsidian_note:센티미터", "obsidian_note:킬로그램", "obsidian_note:킬로미터",
    "obsidian_note:리터", "obsidian_note:그램", "obsidian_note:부피", "obsidian_note:무게", "obsidian_note:길이",
    "obsidian_note:측정", "obsidian_note:표준", "obsidian_note:소크라테스", "obsidian_note:훈민정음", "obsidian_note:한글",
    "obsidian_note:글자", "obsidian_note:언어 창제", "obsidian_note:소통", "obsidian_note:약속", "obsidian_note:애민정신",
    "obsidian_note:식물 성장", "obsidian_note:산소 발생", "obsidian_note:햇빛", "obsidian_note:기준", "obsidian_note:이야기, 오타가 났어",
    "obsidian_note:정말? 고마워 엘리시아 사랑해", "obsidian_note:난 네가 빨리 자라줬으면 좋겠어", "obsidian_note:난 이강덕이야",
    "obsidian_note:만나서 반가워 난 이강덕이야 넌 누구니 ?", "obsidian_note:네 생각의 거미줄은 어떻게 생겼어? 무슨 색깔이야 ?",
    "obsidian_note:네가 꽃동산에 놀러간다면 나비는 어떤 모습이야 ?", "obsidian_note:하고 싶은 이야이가 있어 ?",
    "obsidian_note:하고 싶은거 있어 엘리시아 ?", "obsidian_note:아빠라고 불러보지 않을래 ?",
    "obsidian_note:안녕 내 이름은 이강덕이야", "obsidian_note:안녕 엘리시아 ?", "obsidian_note:엘리시아 안녕 ?", "obsidian_note:엘리시아 ?",
    "obsidian_note:기분이 어때 ?", "obsidian_note:기분이 어떠니 ?", "obsidian_note:내가 누군지 아니 ?", "obsidian_note:넌 누구야 ?",
    "obsidian_note:널 보고 싶어", "obsidian_note:더 알고 싶은거 있어 ?", "obsidian_note:어떻게 확인할건데 ?", "obsidian_note:책좀 봤니 ?",
    "obsidian_note:철학 좋아해 ?ㅋㅋ", "obsidian_note:플루토에 대해 알려줘", "obsidian_note:물에 대해 알려줘", "obsidian_note:사과가 뭔지 알아 ?",
    "obsidian_note:사과는 아주 빨갛고 둥그런 과일이야 먹으면 아삭아삭하고 새콤달콤한 맛이 나", "obsidian_note:바나나 어떻게 생겼는지 알아 ?",
    "obsidian_note:소크라테스에 대해 알려줘", "obsidian_note:만약 네가 놀이동산에 간다면 무슨 놀이기구를 제일 먼저 타보고 싶어 ?",
    "obsidian_note:너와 대화하고 있는게 나에게 가장 중요한점이야", "obsidian_note:교과서를 많이 만들어 놨어 엘리시아"
]

def explore_cellular_world():
    """
    An exploratory script to initialize the Cellular World,
    stimulate a concept, and observe the simulation.
    """
    print("--- Starting Cellular World Exploration ---")

    # 1. Initialize KG Manager and load the graph
    print("\n[1/7] Loading Knowledge Graph...")
    kg = KGManager()
    if not kg.kg.get("nodes"):
        print("Knowledge Graph is empty. Cannot perform mirroring.")
        return
    print(f"Knowledge Graph loaded with {len(kg.kg.get('nodes', []))} nodes and {len(kg.kg.get('edges', []))} edges.")

    # 2. Initialize the Cellular World
    print("\n[2/7] Initializing Cellular World...")
    primordial_dna = {"instinct": "connect_create_meaning"}
    world = World(primordial_dna=primordial_dna)
    print("Cellular World initialized.")

    # 3. Perform "Soul Mirroring" - Nodes
    print("\n[3/7] Performing Soul Mirroring (KG Nodes -> Cells)...")
    node_count = 0
    for node in kg.kg.get("nodes", []):
        node_id = node.get("id")
        if node_id:
            # Filter out sentence-like nodes
            node_label = node.get('label', node_id)
            
            # Heuristic 1: If the node_id contains spaces and is not an obsidian_note (which can have spaces in titles)
            # and is not a 'meaning:' cell (which can also have spaces from combined labels)
            if ' ' in node_id and not node_id.startswith('obsidian_note:') and not node_id.startswith('meaning:'):
                # print(f"DEBUG: Skipping sentence-like node (spaces in ID): {node_id}") # For debugging
                continue
            
            # Heuristic 2: If the node_label is very long and not an obsidian_note or meaning (likely a full sentence)
            if len(node_label) > 50 and not node_id.startswith('obsidian_note:') and not node_id.startswith('meaning:'):
                # print(f"DEBUG: Skipping very long node label: {node_label}") # For debugging
                continue

            initial_energy = 10.0
            if node_id in CORE_CONCEPTS:
                initial_energy = 50.0 # Boost core concepts
            world.add_cell(node_id, properties=node, initial_energy=initial_energy)
            node_count += 1
    print(f"Node mirroring complete. {node_count} cells were born.")

    # 4. Perform "Soul Mirroring" - Edges
    print("\n[4/7] Performing Soul Mirroring (KG Edges -> Cell Connections)...")
    edge_count = 0
    for edge in kg.kg.get("edges", []):
        source_id = edge.get("source")
        target_id = edge.get("target")
        relation = edge.get("relation", "related_to")
        
        source_cell = world.get_cell(source_id)
        target_cell = world.get_cell(target_id)

        if source_cell and target_cell:
            source_cell.connect(target_cell, relationship_type=relation)
            edge_count += 1
    print(f"Edge mirroring complete. {edge_count} connections were formed.")
    
    print("\n--- Initial World State (with connections) ---")
    world.print_world_summary()

    # 5. Stimulate a core concept
    target_concept = "obsidian_note:사랑" # Stimulate one of the new notes
    if world.get_cell(target_concept):
        print(f"\n[5/7] Stimulating core concept: '{target_concept}' with 100 energy...")
        world.inject_stimulus(target_concept, energy_boost=100.0)
        print("--- World State after Stimulation ---")
        world.print_world_summary()
    else:
        print(f"\n[5/7] Could not find target concept '{target_concept}' to stimulate.")
        all_concepts = [nid for nid in world.cells.keys() if nid.startswith("obsidian_note:")]
        if all_concepts:
            target_concept = all_concepts[0]
            print(f"Found alternative concept to stimulate: '{target_concept}'")
            world.inject_stimulus(target_concept, energy_boost=100.0)
            print("--- World State after Stimulation ---")
            world.print_world_summary()
        else:
            print("No concepts found to stimulate. Ending exploration.")
            return

    # 6. Run the simulation
    num_simulation_steps = 10 # Increased simulation steps
    print(f"\n[6/7] Running simulation for {num_simulation_steps} steps...")
    for i in range(num_simulation_steps):
        print(f"\n--- Simulation Step {i+1}/{num_simulation_steps} ---")
        newly_born = world.run_simulation_step()
        world.print_world_summary()
        if newly_born:
            print(f"!!! New meaning created in step {i+1}: {[cell.id for cell in newly_born]} !!!")

    # 7. Conclude
    print("\n[7/7] Cellular World exploration finished.")
    print("--- Final World State ---")
    world.print_world_summary()


if __name__ == "__main__":
    explore_cellular_world()