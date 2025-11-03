from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser, MultiPartParser, FormParser
from rest_framework.response import Response
from app.src.work import verify_claim as vc

@api_view(['POST'])
@parser_classes([JSONParser, MultiPartParser, FormParser])
def verify_claim(request):
    return vc(request)

    # input_type = request.data.get('inputType')
    
    # if input_type == 'image':
    #     image_file = request.FILES.get('file')
    #     # For now we simulate; you can later integrate your image verification logic
    #     result = {
    #         "truthScore": 81,
    #         "verdict": "likely true",
    #         "sources": [
    #             {"name": "Google Reverse Image", "credibility": 90, "url": "https://images.google.com"},
    #             {"name": "TinEye", "credibility": 85, "url": "https://tineye.com"},
    #         ],
    #         "explanation": "Image metadata and reverse search suggest authenticity.",
    #         "aiGenerated": False,
    #         "timestamp": "2025-10-12T12:00:00Z",
    #     }
    # else:
    #     claim = request.data.get('input')
    #     # Simulate text claim verification
    #     result = {
    #         "truthScore": 78,
    #         "verdict": "mixed",
    #         "sources": [
    #             {"name": "Reuters", "credibility": 95, "url": "https://reuters.com"},
    #             {"name": "BBC News", "credibility": 92, "url": "https://bbc.com"},
    #             {"name": "The Guardian", "credibility": 88, "url": "https://theguardian.com"},
    #         ],
    #         "explanation": "This claim contains elements of truth but lacks complete context.",
    #         "aiGenerated": False,
    #         "timestamp": "2025-10-12T12:00:00Z",
    #     }

    # return Response(result)