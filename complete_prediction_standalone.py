# Complete Standalone Brain Tumor Prediction Script
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import matplotlib.pyplot as plt
from PIL import Image
import os

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the Swin Transformer model class (same as in your notebook)
class SwinTransformerClassifier(nn.Module):
    def __init__(self, num_classes=4, model_name='swin_tiny_patch4_window7_224', pretrained=True):
        super(SwinTransformerClassifier, self).__init__()
        
        # Load Swin backbone as a feature extractor with global pooling
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool='avg'
        )
        
        # Determine feature dimension after global pooling
        if hasattr(self.backbone, 'num_features'):
            num_features = self.backbone.num_features
        elif hasattr(self.backbone, 'head') and hasattr(self.backbone.head, 'in_features'):
            num_features = self.backbone.head.in_features
        else:
            raise RuntimeError('Unable to determine feature dimension from backbone')
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits

# Create model instance
print("🔄 Creating model...")
model = SwinTransformerClassifier(num_classes=4, model_name='swin_tiny_patch4_window7_224')
model = model.to(device)

# Load the trained weights
model_path = r"S:\Project\PCOS\archive\best_swin_transformer_model.pth"
print(f"🔄 Loading model from: {model_path}")

try:
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Model loaded successfully!")
    print(f"📊 Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
except FileNotFoundError:
    print(f"❌ Error: Model file not found at {model_path}")
    print("Please check the path and make sure the model file exists.")
    exit()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Define image transformations (same as in your notebook)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_and_visualize_image(image_path, model, transform, device, class_labels):
    """
    Predict and visualize results for a single image
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return None
    
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"✅ Successfully loaded image: {image_path}")
        print(f"📏 Original image size: {image.size}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None
    
    # Prepare image for model
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class_idx = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class_idx].item()
    
    # Prepare results
    predicted_class = class_labels[predicted_class_idx]
    all_probs = {class_labels[i]: prob.item() for i, prob in enumerate(probabilities[0])}
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Display original image
    ax1.imshow(image)
    ax1.set_title(f'Input Image\nSize: {image.size}', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Display prediction results as bar chart
    classes = list(all_probs.keys())
    probabilities_list = list(all_probs.values())
    colors = ['red' if i == predicted_class_idx else 'lightblue' for i in range(len(classes))]
    
    bars = ax2.bar(classes, probabilities_list, color=colors)
    ax2.set_title(f'Prediction Results\nPredicted: {predicted_class}', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Probability')
    ax2.set_ylim(0, 1)
    
    # Add percentage labels on bars
    for bar, prob in zip(bars, probabilities_list):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels for better readability
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print("\n" + "="*60)
    print("🧠 BRAIN TUMOR CLASSIFICATION RESULTS")
    print("="*60)
    print(f"📁 Image Path: {image_path}")
    print(f"🎯 Predicted Class: {predicted_class}")
    print(f"📊 Confidence: {confidence:.2%}")
    print(f"🔍 Model: Swin Transformer")
    
    print(f"\n📈 All Class Probabilities:")
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    for i, (class_name, prob) in enumerate(sorted_probs):
        status = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"   {status} {class_name}: {prob:.2%}")
    
    print("="*60)
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_probabilities': all_probs
    }

# Test with your specific image
print("\n🖼️  Testing with your image...")
image_path = r"S:\Project\PCOS\archive\image.png"  # Your specified image path

# Class labels (same order as in training)
class_labels = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# Run prediction
result = predict_and_visualize_image(
    image_path=image_path,
    model=model,
    transform=val_transforms,
    device=device,
    class_labels=class_labels
)

if result:
    print(f"\n🎉 Prediction completed successfully!")
    print(f"💡 The model predicts: {result['predicted_class']} with {result['confidence']:.1%} confidence")
else:
    print("\n❌ Prediction failed. Please check the image path and try again.")

# Additional functionality: Test with any custom image path
print("\n" + "="*60)
print("🔧 CUSTOM IMAGE TESTING")
print("="*60)
print("To test other images, modify the 'custom_image_path' variable below:")
print()

# Example for testing other images
# custom_image_path = r"S:\Project\PCOS\archive\Training\glioma_tumor\gg (1).jpg"
# custom_result = predict_and_visualize_image(
#     image_path=custom_image_path,
#     model=model,
#     transform=val_transforms,
#     device=device,
#     class_labels=class_labels
# )

print("📝 Instructions:")
print("1. Uncomment the lines above")
print("2. Replace the path with your desired image")
print("3. Run the script to get predictions")
print("\n✨ The model will show both the image and prediction results with confidence scores!")