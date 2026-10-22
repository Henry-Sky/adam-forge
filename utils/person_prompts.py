"""Default photorealistic person prompts for the registration interface.

Override through register_person.py run --prompt / --negative-prompt, or an
explicit --prompt-config JSON containing prompt and negative_prompt strings.
"""

DEFAULT_PROMPT = (
    "Generate a photorealistic real human person in the editable region, "
    "as captured by a live-action camera in the original scene. "
    "Strictly follow the reference image for the person's facial identity, "
    "facial features, hairstyle, clothing, accessories, colors and body proportions. "
    "Preserve visible garment details. Render natural human skin texture, "
    "realistic fabric texture and physically plausible lighting. "
    "Follow the body pose control in every frame, matching joint positions, "
    "gestures, timing and location. Render a solid, fully clothed human body "
    "with natural anatomy, coherent surface detail and smooth temporally "
    "consistent motion. Match scene scale, perspective, lighting and camera "
    "movement. Keep the feet in physically plausible contact with the ground. "
    "Replace the black rectangle and colored skeleton with the person while "
    "preserving the unmasked environment."
)

DEFAULT_NEGATIVE_PROMPT = (
    "colored skeleton, pose lines, black rectangle, stick figure, wireframe, "
    "duplicate person, extra limbs, malformed anatomy, floating body, "
    "transparent body, flickering, changing identity, changing clothing, "
    "blurry character, background changes, animated character, cartoon character, "
    "anime, manga, illustration, cel shading, 2D animation, 3D animation, "
    "CGI, 3D render, stylized character, doll, toy, plastic skin"
)
