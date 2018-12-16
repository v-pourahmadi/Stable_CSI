
function a = feedforwardKeras(image, variance)
    thingSpeakURL = 'http://localhost:5000/estimate_channel';
    %addpath('matlab-json')
    %json.startup
    %data =json.dump(image);
    cel{1} = image;
    cel{2} = variance;
    save('cel.mat','cel')
    fid = fopen('cel.mat','r'); data = uint8(fread(fid));
    %bytes = getByteStreamFromArray(cel);
    %bytes
    options = weboptions('MediaType','application/octet-stream');
    %str = sprintf('%s*%f',mat2str(image), variance); 
    %body = matlab.net.http.MessageBody(bytes);
    %method = matlab.net.http.RequestMethod.POST;
    %obj = matlab.net.http.RequestMessage(method,[], body);
    %complete(obj, thingSpeakURL)
    response = webwrite(thingSpeakURL,matlab.net.base64encode(data),options);
    %response = complete(obj,thingSpeakURL);
    mat = matlab.net.base64decode(response);
    fid = fopen('out.mat','w'); fwrite(fid, mat)
    load out
    a = out;
    %a= eval(response);
    %a = json.load(response);
end