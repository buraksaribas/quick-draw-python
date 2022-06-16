# https://quickdraw.readthedocs.io/en/latest/api.html

from quickdraw import QuickDrawDataGroup
import os

if not os.path.exists('images'):
   os.mkdir('images')
   
with open('categories/categories.txt') as file:
   names = [name.replace('\n', '') for name in file.readlines()]

for name in names:
   images = QuickDrawDataGroup(
       name, 
       recognized=True, 
       max_drawings=1000,
       cache_dir='bin-images', 
       print_messages=False
   )
   name = name.replace(' ', '-')
   path = f'images/{name}/'

   if not os.path.exists(path):
      os.mkdir(path)
      
   for drawing in images.drawings:
      drawing.image.save(f'images/{name}/{drawing.key_id}.jpg')
