
/**
 * Handles operation around the model. Handles the operation around mobilenet, in addition to the second model.
 * Holds a dataset for webcam controls which allows the user to add example Tensors
 * for particular labels. This object will concat them into two large xs and ys.
 */
 class ModelOperator {

  /**
   * @param {number} number of classes the model will learn and predict
   */
  constructor(numClasses) {
    this.numClasses = numClasses;
    this.mobilenet = this.initMobilenet()
  }

  async initMobilenet() {
    const mobilenetDefinition = await tf.loadModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json')
    const outputLayer = mobilenetDefinition.getLayer('conv_pw_13_relu');
    this.mobilenet = tf.model({
        inputs: mobilenetDefinition.inputs,
        outputs: outputLayer.output
    });
  }

  /**
   * Adds an example to the controller dataset.
   * @param {Image} image An image for prediction. It is converted to tensor using mobilenet.
   *                The tensor represents the example. It can be an image,
   *     an activation, or any other type of Tensor.
   * @param {number} label The label of the example. Should be an umber.
   */
  addExample(image, label) {
    const example = this.mobilenet.predict(image);
    // One-hot encode the label.
    const y = tf.tidy(
        () => tf.oneHot(tf.tensor1d([label]).toInt(), this.numClasses));

    if (this.xs == null) {
      // For the first example that gets added, keep example and y so that the
      // ControllerDataset owns the memory of the inputs. This makes sure that
      // if addExample() is called in a tf.tidy(), these Tensors will not get
      // disposed.
      this.xs = tf.keep(example);
      this.ys = tf.keep(y);
    } else {
      const oldX = this.xs;
      this.xs = tf.keep(oldX.concat(example, 0));

      const oldY = this.ys;
      this.ys = tf.keep(oldY.concat(y, 0));

      oldX.dispose();
      oldY.dispose();
      y.dispose();
    }
  }

  mobilenetPredict(img) {
        const predictedClass = tf.tidy(() => {
            const activation = modelOperator.mobilenet.predict(img);
            const predictions = modelOperator.model.predict(activation);
            return predictions.as1D();
        });
        return predictedClass;
  }

  async train() {
    if (this.xs == null) {
      this.updateTrainingStatus && this.updateTrainingStatus('Loading Pre-Trained');
      console.log("No training, loading pre-trained model");
      this.loadPreTrainedModel().then(() => {
        this.updateTrainingStatus && this.updateTrainingStatus('Model Loaded');
      });

      return;
    }

    const xs_data = Array.from(await this.xs.data());
    const xs_shape = this.xs.shape;
    const ys_data = Array.from(await this.ys.data());
    const ys_shape = this.ys.shape;
    const body = {
        'xs': {
            'shape': xs_shape,
            'data': xs_data
        },
        'ys': {
            'shape': ys_shape,
            'data': ys_data
        }
    };
    this.send_mobilenet_data(body);

    this.isTraining = true;
    console.log("Training model on new images")
    // Creates a 2-layer fully connected model. By creating a separate model,
    // rather than adding layers to the mobilenet model, we "freeze" the weights
    // of the mobilenet model, and only train weights from the new model.
    this.model = tf.sequential({
      layers: [
        // Flattens the input to a vector so we can use it in a dense layer. While
        // technically a layer, this only performs a reshape (and has no training
        // parameters).
        tf.layers.flatten({inputShape: [7, 7, 256]}),
        // Layer 1
        tf.layers.dense({
          units: tsParams.units,
          activation: 'relu',
          kernelInitializer: 'varianceScaling',
          useBias: true
        }),
        // Layer 2. The number of units of the last layer should correspond
        // to the number of classes we want to predict.
        tf.layers.dense({
          units: 2,
          kernelInitializer: 'varianceScaling',
          useBias: false,
          activation: 'softmax'
        })
      ]
    });

    // Creates the optimizers which drives training of the model.
    const optimizer = tf.train.adam(tsParams.learningRate);
    const { model } = this;
    // We use categoricalCrossentropy which is the loss function we use for
    // categorical classification which measures the error between our predicted
    // probability distribution over classes (probability that an input is of each
    // class), versus the label (100% probability in the true class)>
    model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

    // We parameterize batch size as a fraction of the entire dataset because the
    // number of examples that are collected depends on how many examples the user
    // collects. This allows us to have a flexible batch size.
    const batchSize =
        Math.floor(this.xs.shape[0] * tsParams.batchSize);
    if (!(batchSize > 0)) {
      throw new Error(
          `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
    }

    // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
      model.fit(this.xs, this.ys, {
      batchSize,
      epochs: tsParams.epochs,
      callbacks: {
        onBatchEnd: async (batch, logs) => {
          this.updateTrainingStatus && this.updateTrainingStatus('Loss: ' + logs.loss.toFixed(5));
          await tf.nextFrame();
        }
      }
    });

    this.isTraining = false;
  }

  async send_mobilenet_data(data) {
    const url = '/upload_mobilenet_pred';
    let headers = new Headers();
    headers.append('Content-Type', 'application/json');

    const myInit = {
        method: 'POST'
        , headers: headers
        , body: JSON.stringify(data)
    };
    fetch(url, myInit)

  }

  async loadPreTrainedModel() {
        const trained_model = await tf.loadModel('static/spaceinvaders/model/model.json')
        const output_layer = trained_model.getLayer('output_softmax');
        this.model = tf.model({
            inputs: trained_model.inputs,
            outputs: output_layer.output
        });
        console.log("Loaded pre-trained model successfully.")
    }
}