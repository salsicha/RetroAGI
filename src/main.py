"""Main script to run the agent."""
import cv2
import torch
from torchvision import transforms

from src.models.occipital import OccipitalLobe
from src.models.temporal import TemporalLobe
from src.utils.screen_capture import ScreenCapture


def main():
    """Main function to run the agent."""
    screen_capturer = ScreenCapture()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    occipital_lobe = OccipitalLobe(latent_dim=128).to(device)
    temporal_lobe = TemporalLobe(latent_dim=128, hidden_dim=128, vocab_size=10).to(device)

    # For now, let's just create a dummy optimizer and loss function
    # as we are not training the models yet.
    optimizer_o = torch.optim.Adam(occipital_lobe.parameters(), lr=0.001)
    optimizer_t = torch.optim.Adam(temporal_lobe.parameters(), lr=0.001)
    criterion_o = torch.nn.MSELoss()
    criterion_t = torch.nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    try:
        while True:
            # Capture the screen
            screen = screen_capturer.capture_screen()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2RGB)

            # Preprocess the image
            img_tensor = transform(screen).unsqueeze(0).to(device)

            # Occipital Lobe: Process visual input
            reconstructed_tensor = occipital_lobe(img_tensor)
            latent_vector = occipital_lobe.get_latent(img_tensor)
            
            # Temporal Lobe: Generate description
            generated_sequence = temporal_lobe(latent_vector)
            generated_text = temporal_lobe.sequence_to_text(generated_sequence.squeeze())
            print(f"Generated Text: {generated_text}")

            # Postprocess the reconstructed image
            reconstructed_img = reconstructed_tensor.squeeze(0).cpu().detach().numpy()
            reconstructed_img = reconstructed_img.transpose(1, 2, 0)
            reconstructed_img = (reconstructed_img * 255).astype('uint8')


            # Display the original and reconstructed images
            cv2.imshow("Original", cv2.cvtColor(screen, cv2.COLOR_RGB2BGR))
            cv2.imshow("Reconstructed", cv2.cvtColor(reconstructed_img, cv2.COLOR_RGB2BGR))


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()