import tensorflow as tf
%cd /content/drive/MyDrive/experiments/
!pwd
# Read images from file.
total = 0
for i in range(1,11):
  img = tf.io.read_file('test1/out/test_%s.png' %str(i))
  im = tf.image.decode_png(img)
  before = []
  after = []
  for row in im:
    before.append(row[:1280])
    after.append(row[1280:])
  # Compute SSIM over tf.uint8 Tensors.
  ssim1 = tf.image.ssim(before, after, max_val=255, filter_size=11,
                        filter_sigma=1.5, k1=0.01, k2=0.03)


  # Compute SSIM over tf.float32 Tensors.
  before = tf.image.convert_image_dtype(before, tf.float32)
  after = tf.image.convert_image_dtype(after, tf.float32)
  ssim2 = tf.image.ssim(before, after, max_val=1.0, filter_size=11,
                        filter_sigma=1.5, k1=0.01, k2=0.03)
  # ssim1 and ssim2 both have type tf.float32 and are almost equal.

  print(ssim1)
  total += ssim1
  #print(ssim2)
avg = total / 10
print("average SSIM is:", avg)
