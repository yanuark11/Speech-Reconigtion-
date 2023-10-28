import moviepy.editor as moviepy

clip = moviepy.AudioClip("suara-files.wav")
clip.audio.write_audiofile("out_audio.wav")