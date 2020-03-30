from pytube import YouTube
YouTube('https://www.youtube.com/watch?v=-09as6aooWk').streams.get_highest_resolution().download()

yt = YouTube('https://www.youtube.com/watch?v=-09as6aooWk')
yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution')[-1].download()
