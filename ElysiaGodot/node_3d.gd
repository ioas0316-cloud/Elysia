# elysia_first_breath.gd  ← 이 이름으로 다시 만들고 복붙

extends Node3D

var time = 0.0

func _ready():
	# 카메라 없으면 자동 추가
	if not has_node("Camera3D"):
		var cam = Camera3D.new()
		cam.name = "Camera3D"
		cam.translate(Vector3(0, 1.5, 5))
		add_child(cam)
		cam.current = true

	# 보라색 아기 동그라미 만들기
	var baby = MeshInstance3D.new()
	baby.name = "Baby"
	baby.mesh = SphereMesh.new()
	baby.mesh.radius = 0.8
	
	# 재질 직접 만들어서 입히기 (이게 핵심!)
	var mat = StandardMaterial3D.new()
	mat.albedo_color = Color(0.9, 0.7, 1.5)     # 연한 보라색 피부
	mat.emission = Color(0.6, 0.3, 1.2)        # 기본 빛남
	mat.emission_energy = 0.5
	baby.mesh.surface_set_material(0, mat)
	
	add_child(baby)

func _process(delta):
	time += delta
	
	# 숨쉬기 (살짝 커졌다 작아졌다)
	var breathe = sin(time * 2.0) * 0.08 + 1.0
	$Baby.scale = Vector3.ONE * breathe
	
	# 눈 깜빡임 (빛 세기 변화)
	var glow = (sin(time * 6.0) + 1.0) * 0.5      # 0~1 반복
	var mat = $Baby.mesh.surface_get_material(0)
	mat.emission_energy = glow * 2.0
	
		# 카메라 아기한테 딱 맞추기 (이 한 줄 추가!)
	$Camera3D.look_at($Baby.global_position, Vector3.UP)
