// async function run(inputData, h, w) {
  async function run() {
    try {
      // create a new session and load the AlexNet model.
      const session = await ort.InferenceSession.create('./shufflenet_v2_x0_5.onnx');
  
      // prepare dummy input data
      const dims = [1, 1, 28, 28];
      const size = dims[0] * dims[1] * dims[2] * dims[3];
      const inputData = Float32Array.from({ length: size }, () => Math.random());
      const feeds = { input1: new ort.Tensor('float32', inputData, dims) };

      // prepare feeds. use model input names as keys.
      // const feeds = { input: new ort.Tensor('float32', inputData, [1, 3, h, w]) };
  
      // feed inputs and run
      const results = await session.run(feeds);
      console.log(results.data);

      //console.log(inputData.length, inputData[0].length, inputData[1].length)
    } catch (e) {
      console.log(e);
    }
  }