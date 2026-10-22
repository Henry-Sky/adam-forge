# 写实人物提示词

使用用户选择的实际参考图，不把某一次的身份、制服或动作写成通用要求。外观附件以图中实际可见内容为准。下面正向避免将人物称为动画角色；“动作控制”仍由 FBX 姿态提供。

## Positive

Generate a photorealistic real human person in the editable region, as captured by a live-action camera in the original scene. Strictly follow the reference image for the person's facial identity, facial features, hairstyle, clothing, accessories, colors and body proportions. Preserve visible garment details. Render natural human skin texture, realistic fabric texture and physically plausible lighting. Follow the body pose control in every frame, matching joint positions, gestures, timing and location. Render a solid, fully clothed human body with natural anatomy, coherent surface detail and smooth temporally consistent motion. Match scene scale, perspective, lighting and camera movement. Keep the feet in physically plausible contact with the ground. Replace the black rectangle and colored skeleton with the person while preserving the unmasked environment.

## 正向中文

在可编辑区域生成照片级写实真人，使其看起来由原场景中的实拍相机拍摄。严格按照参考图保留人物身份、面部特征、发型、服装、配饰、颜色及身体比例，保留可见的服装细节。呈现自然的人体皮肤纹理、真实织物纹理和符合物理规律的光照。每帧遵循身体姿态控制，匹配关节位置、手势、动作时序和位置。生成人体结构自然、衣着完整、实体清楚的人物，保持表面细节连贯、运动平滑和时间一致性。匹配场景尺度、透视、光照及相机运动，使脚部与地面的接触合理。以人物替换黑色矩形和彩色骨架，同时保留未被遮罩覆盖的环境。

## Negative

colored skeleton, pose lines, black rectangle, stick figure, wireframe, duplicate person, extra limbs, malformed anatomy, floating body, transparent body, flickering, changing identity, changing clothing, blurry character, background changes, animated character, cartoon character, anime, manga, illustration, cel shading, 2D animation, 3D animation, CGI, 3D render, stylized character, doll, toy, plastic skin

## 负向中文

彩色骨架、姿态线条、黑色矩形、火柴人、线框、重复人物、多余肢体、畸形人体结构、悬浮身体、透明身体、闪烁、人物身份变化、服装变化、人物模糊、背景变化、动画人物、卡通人物、动漫、漫画、插画、赛璐璐着色、二维动画、三维动画、计算机生成画面、三维渲染、风格化角色、玩偶、玩具、塑料皮肤。

参考图保真受模型能力、人物有效像素、姿态跨度和控制类型影响；这些指令不构成身份或外观完全一致的保证。若用户要求卡通等风格，相应调整正负向，避免互相冲突。
