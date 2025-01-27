function toggleStocHyp() {
    const button = document.getElementById('toggleButton');
    const imagesHyp = document.querySelectorAll('img.hyp');
    const imagesHypStoc = document.querySelectorAll('img.hyp_stoc');
    const isDeterministic = button.getAttribute('data-state') === 'deterministic';

    // Toggle visibility of images
    imagesHyp.forEach(img => img.style.display = isDeterministic ? 'none' : 'block');
    imagesHypStoc.forEach(img => img.style.display = isDeterministic ? 'block' : 'none');

    // Update button state and label
    button.setAttribute('data-state', isDeterministic ? 'stochastic' : 'deterministic');
    button.textContent = isDeterministic ? 'Show Deterministic Hypotheses' : 'Show Stochastic Hypotheses';
}

function toggleExpandable(id) {
    const element = document.getElementById(id);
    if (element.style.display === "none" || element.style.display === "") {
        element.style.display = "block";
    } else {
        element.style.display = "none";
    }
}


document.addEventListener('DOMContentLoaded', () => {
    const overlay = document.getElementById('imageOverlay');
    const overlayImage = document.getElementById('overlayImage');
    let currentImages = [];
    let currentIndex = 0;

    // Open overlay
    document.querySelectorAll('.model img').forEach((img, index) => {
        img.addEventListener('click', (e) => {
            currentImages = Array.from(img.closest('.models').querySelectorAll('img'));
            currentIndex = currentImages.indexOf(e.target);
            showImage(currentIndex);
        });
    });

    // Show image in overlay
    function showImage(index) {
        overlayImage.src = currentImages[index].src;
        overlay.style.display = 'flex';
    }

    // Close overlay
    function closeOverlay() {
        overlay.style.display = 'none';
    }

    // Navigate with arrow keys
    document.addEventListener('keydown', (e) => {
        if (overlay.style.display === 'flex') {
            if (e.key === 'ArrowRight') {
                currentIndex = (currentIndex + 1) % currentImages.length;
                showImage(currentIndex);
            } else if (e.key === 'ArrowLeft') {
                currentIndex = (currentIndex - 1 + currentImages.length) % currentImages.length;
                showImage(currentIndex);
            } else if (e.key === 'Escape') {
                closeOverlay();
            }
        }
    });

    // Expose closeOverlay globally
    window.closeOverlay = closeOverlay;
});
