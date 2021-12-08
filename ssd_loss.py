import tensorflow as tf

class CustomLoss(object):
    def __init__(self, neg_pos_ratio, loc_loss_alpha):
        self.neg_pos_ratio = tf.constant(neg_pos_ratio, dtype=tf.float32)
        self.loc_loss_alpha = tf.constant(loc_loss_alpha, dtype=tf.float32)

    def loc_loss_fn(self, actual_deltas, pred_deltas):
        """Calculating SSD localization loss value for only positive samples.
        inputs:
            actual_deltas = (batch_size, total_prior_boxes, [delta_y, delta_x, delta_h, delta_w])
            pred_deltas = (batch_size, total_prior_boxes, [delta_y, delta_x, delta_h, delta_w])

        outputs:
            loc_loss = localization / regression / bounding box loss value
        """
        # Localization / bbox / regression loss calculation for all bboxes
        loc_loss_fn = tf.losses.Huber(reduction=tf.losses.Reduction.NONE)
        loc_loss_for_all = loc_loss_fn(actual_deltas, pred_deltas)
        # After tf 2.2.0 version, the huber calculates mean over the last axis
        loc_loss_for_all = tf.cond(tf.greater(tf.rank(loc_loss_for_all), tf.constant(2)),
                                   lambda: tf.reduce_sum(loc_loss_for_all, axis=-1),
                                   lambda: loc_loss_for_all * tf.cast(tf.shape(pred_deltas)[-1], dtype=tf.float32))
        #
        pos_cond = tf.reduce_any(tf.not_equal(actual_deltas, tf.constant(0.0)), axis=2)
        pos_mask = tf.cast(pos_cond, dtype=tf.float32)
        total_pos_bboxes = tf.reduce_sum(pos_mask, axis=1)
        #
        loc_loss = tf.reduce_sum(pos_mask * loc_loss_for_all, axis=-1)
        total_pos_bboxes = tf.where(tf.equal(total_pos_bboxes, tf.constant(0.0)), tf.constant(1.0), total_pos_bboxes)
        loc_loss = loc_loss / total_pos_bboxes
        #
        return loc_loss * self.loc_loss_alpha

    def conf_loss_fn(self, actual_labels, pred_labels, actual_ages, pred_ages):
        """Calculating SSD confidence loss value by performing hard negative mining as mentioned in the paper.
        inputs:
            actual_labels = (batch_size, total_prior_boxes, total_labels)
            pred_labels = (batch_size, total_prior_boxes, total_labels)
            actual_ages = (batch_size, total_prior_boxes, total_ages)
            pred_ages = (batch_size, total_prior_boxes, total_ages)

        outputs:
            conf_loss = confidence / class / label and age loss value
        """
        # Confidence / Label loss calculation for all labels
        conf_loss_fn = tf.losses.CategoricalCrossentropy(reduction=tf.losses.Reduction.NONE)
        conf_loss_for_all = conf_loss_fn(actual_labels, pred_labels)
        conf_loss_for_all_1 = conf_loss_fn(actual_ages, pred_ages)
        #
        pos_cond = tf.reduce_any(tf.not_equal(actual_labels[..., 1:], tf.constant(0.0)), axis=2)
        pos_mask = tf.cast(pos_cond, dtype=tf.float32)
        total_pos_bboxes = tf.reduce_sum(pos_mask, axis=1)
        pos_cond_1 = tf.reduce_any(tf.not_equal(actual_ages[..., 1:], tf.constant(0.0)), axis=2)
        pos_mask_1 = tf.cast(pos_cond_1, dtype=tf.float32)
        total_pos_bboxes_1 = tf.reduce_sum(pos_mask_1, axis=1)
        # Hard negative mining
        total_neg_bboxes = tf.cast(total_pos_bboxes * self.neg_pos_ratio, tf.int32)
        total_neg_bboxes_1 = tf.cast(total_pos_bboxes_1 * self.neg_pos_ratio, tf.int32)
        #
        masked_loss = conf_loss_for_all * actual_labels[..., 0]
        masked_loss_1 = conf_loss_for_all_1 * actual_ages[..., 0]
        sorted_loss = tf.argsort(masked_loss, direction="DESCENDING")
        sorted_loss = tf.argsort(sorted_loss)
        sorted_loss_1 = tf.argsort(masked_loss_1, direction="DESCENDING")
        sorted_loss_1 = tf.argsort(sorted_loss_1)
        neg_cond = tf.less(sorted_loss, tf.expand_dims(total_neg_bboxes, axis=1))
        neg_cond_1 = tf.less(sorted_loss_1, tf.expand_dims(total_neg_bboxes_1, axis=1))
        neg_mask = tf.cast(neg_cond, dtype=tf.float32)
        neg_mask_1 = tf.cast(neg_cond_1, dtype=tf.float32)
        #
        final_mask = pos_mask + neg_mask
        final_mask_1 = pos_mask_1 + neg_mask_1
        conf_loss = tf.reduce_sum(final_mask * conf_loss_for_all, axis=-1)
        conf_loss_1 = tf.reduce_sum(final_mask_1 * conf_loss_for_all_1, axis=-1)
        total_pos_bboxes = tf.where(tf.equal(total_pos_bboxes, tf.constant(0.0)), tf.constant(1.0), total_pos_bboxes)
        total_pos_bboxes_1 = tf.where(tf.equal(total_pos_bboxes_1, tf.constant(0.0)), tf.constant(1.0), total_pos_bboxes_1)
        conf_loss = conf_loss / total_pos_bboxes
        conf_loss_1 = conf_loss_1 / total_pos_bboxes_1
        #
        return ((conf_loss+conf_loss_1)/2)
