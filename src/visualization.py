def print_CILP_results(epoch, loss, logits_per_img, is_train=True):
    if is_train:
        print(f"Epoch {epoch}")
        print(f"Train Loss: {loss} ")
    else:
        print(f"Valid Loss: {loss} ")
    print("Similarity:")
    print(logits_per_img)