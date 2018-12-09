const cv = require('opencv4nodejs');
const fs = require('fs');
const path = require('path');
const tasks  = require('./commons.js');
const utils = require('./utils');
var express = require('express');
var fileUpload = require('express-fileupload');
//		 csvdata = require('csvdata'),
//		  jsonexport = require('jsonexport'),
//		  diskspace = require('diskspace'),
var request = require('request');
// var poster =  require("poster");
var cors = require('cors');
var events = require('events');
var multer = require('multer');
//var bodyParser = require('body-parser');
//var xFrameOptions = require('x-frame-options');

//constructors
var eventEmitter = new events.EventEmitter();
var app = express();

//fs vars
var lastImageName;
var options = {
    root: __dirname + '/',
//    dotfiles: 'deny',
    headers: {
        'x-timestamp': Date.now(),
        'x-sent': true,
		  'X-Frame-Options': 'ALLOW-FROM *'
    }
};

//use
app.use(cors({origin: '*'}));
//app.use(fileUpload());


/////
//FS
/////
//img filter
const imageFilter = function (req, file, cb) {
    // accept image only
    if (!file.originalname.match(/\.(jpg|jpeg|png|gif)$/)) {
        return cb(new Error('incompatible encoding, please try uploading the image in a .png format'), false);
    }
    cb(null, true);
};
//file upload storage
var Storage = multer.diskStorage({
     destination: function(req, file, callback) {
//		 			console.log("request: " + JSON.stringify(file.originalname));
			 if(file){
			 	var now = new Date();
			 lastImageName = file.fieldname + "_" + now.getDate() + "_" + now.getMonth() + "_" + now.getFullYear() + "_" + now.getHours() + "_" + now.getSeconds() + "_" + now.getMilliseconds() + "_" + file.originalname;
			 	console.log("file name:" + lastImageName);
         }
				 
				 callback(null, "./www/j-finder.com/uploads");
     },
   filename: function(req, file, callback) {
         callback(null, lastImageName);
     }
 });
const upload = multer({ storage: Storage ,fileFilter: imageFilter}).single('img');


////
//DNN
////

if (!cv.xmodules.dnn) {
  throw new Error('exiting: opencv4nodejs compiled without dnn module');
}

// replace with path where you unzipped inception model
const inceptionModelPath = './dnn/inception5h';
//const uploadPath = './www/j-finder.com/uploads';

//const modelFile = path.resolve(inceptionModelPath, 'tensorflow_inception_graph.pb');
//const classNamesFile = path.resolve(inceptionModelPath, 'imagenet_comp_graph_label_strings.txt');
const modelFile = path.resolve(inceptionModelPath, 'graph_v2.pb');
const classNamesFile = path.resolve(inceptionModelPath, 'test_classes.txt');
if (!fs.existsSync(modelFile) || !fs.existsSync(classNamesFile)) {
  console.log('could not find inception model');
  console.log('download the model from: https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip');
  throw new Error('exiting');
}

// read classNames and store them in an array
const classNames = fs.readFileSync(classNamesFile).toString().split('\n');

// initialize tensorflow inception model from modelFile
const net = cv.readNetFromTensorflow(modelFile);

const classifyImg = (img) => {
  //  model works with 150 x 150 images, so we resize
  // our input images and pad the image with white pixels to
  // make the images have the same width and height
//  const maxImgDim = 224;
//  const white = new cv.Vec(255, 255, 255);
//  const imgResized = img.resizeToMax(maxImgDim).padToSquare(white);
  
  //smaller img input model
  let imgResized = img.resize(150,150);
  imgResized = imgResized.bgrToGray();
  //large img input model
//  const imgResized = img.resize(224,224);
//  .padToSquare(white);

  // network accepts blobs as input
  const inputBlob = cv.blobFromImage(imgResized);
  net.setInput(inputBlob);

  // forward pass input through entire network, will return
  // classification result as 1xN Mat with confidences of each class
  const outputBlob = net.forward();
	
	for(i=0; i<outputBlob.cols; i++){
	console.log("outputBlob at " + i + ": " + outputBlob.at(0, i));
	
	}

//	console.log("outputBlob:" + JSON.stringify(outputBlob, null, 2));
//	console.log("outputBlob class:" + outputBlob.at(0));
//
//	console.log("outputBlob value:" + outputBlob.at(0));
//	console.log("outputBlob cols:" + outputBlob.cols);
//	console.log("outputBlob rows:" + outputBlob.rows);
	
	
  // find all labels with a minimum confidence
  const minConfidence = 0.05;
  const locations =
    outputBlob
      .threshold(minConfidence, 1, cv.THRESH_TOZERO)
      .convertTo(cv.CV_8U)
      .findNonZero();
		
//		if(locations != null && locations.length > 1){
////		console.log("locations: " + JSON.stringify(locations,null,2));
//		if(locations[0].confidence === locations[1].confidence){
//			return ['Looks mixed with ' + locations[1].confidence + '% confidence'];
////			return ['Looks mixed with ' + (locations[1].confidence - Math.floor(Math.random() * 1000)/1000) + '% confidence'];
//		}
//		}
		console.log("locations: " + JSON.stringify(locations));
	  const result =
    locations.map(pt => ({
      confidence: parseFloat(outputBlob.at(0, pt.x) * 100) ,
      className: classNames[pt.x]
    }))	 
      // sort result by confidence
      .sort((r0, r1) => r1.confidence - r0.confidence)
		.map(res => `Looks ${res.className} with ${res.confidence}% confidence`);
//      .map(res => `Looks ${res.className} with ${res.confidence - (Math.floor(Math.random() * 1000)/1000)}% confidence`);	
		
		return result;

}

////
//face detection
////

const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);
let whiteMat = new cv.Mat(100, 100,cv.CV_8UC3, [50, 50, 50])
const webcamPort = 0;
let noFaces = true;

function detectFaces(img) {
  // restrict minSize and scaleFactor for faster processing
  const options = {
    minSize: new cv.Size(100, 100),
    scaleFactor: 1.2,
    minNeighbors: 10
  };
  return classifier.detectMultiScaleGpu(img.bgrToGray(), options).objects;
}


function calcMovement(faceIndex){
	if(displayedRects[faceIndex] == null){
		displayedRects[faceIndex] = new cv.Rect();
	}

}

function faceRects(faces, img){
//img = img.flip(1);

if(faces.length){
	
//	img = img.flip(1);
//	console.log("facerects: "+ JSON.stringify(faces,null,2));
//	face = img.getRegion(new cv.Rect(faces[0].x,faces[0].y,faces[0].width,faces[0].height) );
//	img = img.add(whiteMat);
//	img = img.gaussianBlur(new cv.Size(15, 15), 30.2)
//	face.copyTo(img.getRegion(new cv.Rect(faces[0].x,faces[0].y,faces[0].width,faces[0].height)));
//	
	faces.forEach((face, index)=>{
		
		console.log("face idx: " + index);
//		calcMovement(index);
		
		utils.drawGrayRect(img, face);
		whiteMat = whiteMat.resize(face.width,face.height);
		_face = img.getRegion(new cv.Rect(face.x,face.y,face.width,face.height));
		clone = _face.bgrToGray().cvtColor(cv.COLOR_GRAY2BGR);
		_face = _face.addWeighted(0,clone,1,0);
		_face.copyTo(img.getRegion(new cv.Rect(face.x,face.y,face.width,face.height)));
		
//			img = img.flip(1);
	////
	//run dnn prediction
	////
	
		input = _face.resize(150,150);
		input = input.bgrToGray();
//cv.imshow('jewishFinder', clone);;
		
		const predictions = classifyImg(clone);
//		console.log("prediction length: " + predictions.length);
//	  	predictions.forEach(p => console.log("p:" +p + "\nprediction length: " + predictions.length ));
		
		console.log("predictions: " + JSON.stringify(predictions, null,2));
		
		const alpha = 0.4;
		cv.drawTextBox(
		 img,
		 { x:  face.x , y: face.y -32 },
		 predictions.map(p => ({ text: p, fontSize: 0.5, thickness: 1 })),
		 alpha
		);


	});
}
//return img;
//cv.imshow('jewishFinder', img);
}

//root-home route
//app.get('/', function (req, res) {
//	//res.send("response..: Success!! ");
//	res.send("jFinder server");
//	logRequest(req.protocol + '://' + req.get('host'),req.originalUrl,req.ip);
//});  
 

//file upload
app.post('/upload',  function (req, res, callback) {
		console.log("requeset: ");

		upload(req, res, function (err) {
		

		if (err){
		
      console.log('upload error: ' + JSON.stringify(err));
//      res.status(400).send("fail saving image");
    } else {
//		callback(console.log('callback: ' + lastImageName));
//		callback(console.log('image uploaded to: ' +lastImageName + '\n' + 
//													'from (client service location): ' + req.ip));
		console.log('./www/j-finder.com/uploads/' + lastImageName);
		mat = cv.imread('./www/j-finder.com/uploads/' + lastImageName) ;
		mat = mat.resizeToMax(900);
//		cv.imshow('jewishFinder', mat);
		
		faces = detectFaces(mat);
		
		//check if no faces
		(faces.length  == 0 ) ? noFaces = true : noFaces = false; 
		
		faceRects(faces, mat);
		cv.imwrite('./www/j-finder.com/uploads/' + lastImageName, mat);
//		res.send(lastImageName);
		res.writeHead(200, {'Content-Type': 'text/plain', 
									'Access-Control-Allow-Origin': '*',
									'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
									'Access-Control-Allow-Headers': 'DNT,X-CustomHeader,Keep-Alive,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type'});
		res.write(  lastImageName + ',' + noFaces.toString() );
		res.end();

		}
		
			
    		
//	console.log("req.file.originalname: " + lastImagePath.file);
//    res.redirect('/');
});
});

//serve static
app.use(express.static('www/j-finder.com'));

function logRequest(requestURL, requestQuary, requestIp){
	now = new Date();
	var zeroFroHours = '';
	console.log("\nnew request:\n" + requestURL +  requestQuary );
	(now.getHours() != (10 || 11 || 12) ) ? zeroFroHours = "0" : zero = "";		
	console.log( now.getMonth() + "/" + now.getDate() + "/" + now.getFullYear() +
							"\n" + zeroFroHours + now.getHours() + ":" + now.getMinutes() + ":" + now.getSeconds() +
							"\nfrom:\n(client service location)\n" + requestIp + 
							"\n - process done - \n");
}

var ip = process.env.IP || '127.0.0.1';
var port = process.env.PORT || '3003';

app.listen(port, () => console.log('running on port 3003'));
	console.log('\033[2J');
  console .log(new Date());
  console .log(' Hi there, Welcome to jFinder API server');
  
   console.log('________________________________________________________');
  

