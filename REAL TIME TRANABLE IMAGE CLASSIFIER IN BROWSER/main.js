let model;
let mobilenet;

let class1 = 0;
let class2 = 0;

let webcam = new Webcam(document.getElementById('videoElement'));

let dataset = new dataSet();

let isPredict = false;


async function loadMobilenet() {
    const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    const layer = mobilenet.getLayer('conv_pw_13_relu');
    // console.log(1212)
    return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
}


async function train() {
    document.getElementById('ontrain').innerHTML = "Training....";
    dataset.oneHotEncode(2); //for two classes
    model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: mobilenet.outputs[0].shape.slice(1) }));
    // model.add(tf.layers.dense({ units: 200, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 2, activation: 'softmax' }));

    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['acc'] });

    await model.fit(dataset.xs, dataset.ys,
        {
            epoch: 10,
            callbacks: {
                onBatchEnd: async (batch, logs) => {
                    loss = logs.loss.toFixed(5);
                    console.log('LOSS: ' + loss);
                },
                // onEpochEnd: async (epoch, logs) => {
                //     console.log("epoch: " + epoch + ' loss: ' + logs.loss);
                // }
            }
        }
    )
    document.getElementById("predictButton").style.display = 'inline-block';
    document.getElementById('ontrain').innerHTML = "Done....";
    alert("Training is done!");
    console.log("Done")
}

function handleButton(e) {
    if (e.id == 0) {
        class1++;
        //docunment...
        document.getElementById("c1").innerHTML = class1;
    } else {
        class2++;
        //document
        document.getElementById("c2").innerHTML = class2;

    }
    let lebel = parseInt(e.id);
    const image = webcam.capture();
    dataset.add(mobilenet.predict(image), lebel);
}

async function predict() {

    while (isPredict) {
        let prediction = tf.tidy(() => {
            let img = webcam.capture();
            let mobileNetPrediction = mobilenet.predict(img);
            return model.predict(mobileNetPrediction);;
        })
        // console.log(prediction.data()[0]);
        const classId = (await prediction.as1D().argMax().data())[0];
        // console.log(prediction.dataSync());
        let class1Prob = Math.floor(prediction.dataSync()[0] * 100);
        let class2Prob = Math.floor(prediction.dataSync()[1] * 100);

        document.getElementById("result").innerHTML = classId == 0 ? "class 1" : "class 2";
        document.getElementById("c1bar").style.width = class1Prob + "%";
        document.getElementById("c2bar").style.width = class2Prob + "%";
    }

}

function prediction() {
    isPredict = true;
    document.getElementById('bars').style.display = 'block';
    document.getElementById('ontrain').innerHTML = "";
    predict();
}


async function init() {
    await webcam.setup();
    mobilenet = await loadMobilenet();
    tf.tidy(() => mobilenet.predict(webcam.capture()));
}



init();