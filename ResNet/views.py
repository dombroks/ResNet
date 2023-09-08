from rest_framework.views import APIView
from rest_framework.response import Response
from PIL import Image
from ResNet import resnet50
import base64
from io import BytesIO


class ImagePredictionAPIView(APIView):
    def post(self, request, format=None):
        # Retrieve the base64-encoded image from the request data
        encoded_image = request.data.get('image')

        if encoded_image:
            # Decode the base64-encoded image
            decoded_image = base64.b64decode(encoded_image.encode())

            # Load the image using PIL
            image = Image.open(BytesIO(decoded_image))

            # Resize the image to (224, 224)
            image = image.resize((224, 224))

            # Perform prediction and get the result
            decoded_prediction, score = resnet50.predict(image)

            # Return the prediction result as a JSON response
            return Response({
                'prediction': decoded_prediction,
                'score': score
            })
        else:
            return Response({'error': 'Could not process the image'}, status=400)
