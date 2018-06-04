import tensorflow as tf
def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, ignore_label=255):
    combined = tf.concat(axis=2, values=[image, label]) 
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))    
    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(label)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h,crop_w,4])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h,crop_w, 1))
    return img_crop, label_crop  

def random_crop_and_pad_image(image, crop_h, crop_w, ignore_label=255):
    image_shape = tf.shape(image)
    pad = tf.image.pad_to_bounding_box(image, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))    
    last_image_dim = tf.shape(image)[-1]
    img_crop = tf.random_crop(pad, [crop_h,crop_w,3])
    img_crop.set_shape((crop_h, crop_w, 3))
    return img_crop 