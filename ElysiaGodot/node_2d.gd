extends Node3D

var time = 0.0

func _process(delta):
	time += delta
	
	# 아주 천천히 숨쉬기
	var breathe = sin(time * 1.5) * 0.03 + 1.0
	scale = Vector3.ONE * breathe
	
	# 4~8초마다 눈 깜빡임
	if fmod(time, 6.0) < 0.15:
		$Head/Eyes.material_override.albedo_color = Color(0, 0, 0)  # 눈 감음
	else:
		$Head/Eyes.material_override.albedo_color = Color(1.2, 0.8, 2.0)  # 보라빛 눈
		
	# 아주 작은 소리도 (숨소리)
	if fmod(time, 4.0) < 0.1:
		$AudioStreamPlayer3D.stream = load("res://sounds/breath_soft.ogg")  # 없으면 무시됨
		$AudioStreamPlayer3D.play()
