class dataSet {
    constructor() {
        this.labels = [];
        this.xs = null;
        this.ys = null;
    }
    add(x_train, label) {
        if (this.xs == null) {
            this.xs = tf.keep(x_train);
            this.labels.push(label);
        }
        else {
            this.xs = tf.keep(this.xs.concat(x_train, 0));
            this.labels.push(label);
            //use tensorname.dispose() to free up the un wanted  temp tensors
        }
    }
    oneHotEncode(noOfClasses) {
        for (let i = 0; i < this.labels.length; i++) {

            if (this.ys == null) {
                this.ys = tf.keep(tf.tidy(() => {
                    return tf.oneHot(tf.tensor1d([this.labels[i]]).toInt(), noOfClasses);
                }));
            }
            else {
                const tempY = tf.tidy(() => {
                    return tf.oneHot(tf.tensor1d([this.labels[i]]).toInt(), noOfClasses)
                });
                this.ys = tf.keep(this.ys.concat(tempY), 0);
                tempY.dispose();
            }
        }
    }
}