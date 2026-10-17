import { useEffect, useRef } from "react";

interface Props {
  src: string;
  caption: string;
  onClose: () => void;
}

/** Простое модальное окно для увеличенного кадра. Esc и клик по фону закрывают. */
export function ImageModal({ src, caption, onClose }: Props) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const prevFocus = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      } else if (e.key === "Tab") {
        // Единственный фокусируемый элемент — кнопка закрытия.
        e.preventDefault();
        closeRef.current?.focus();
      }
    }
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
      prevFocus?.focus?.();
    };
  }, [onClose]);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label={caption}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-head">
          <span className="modal-caption">{caption}</span>
          <button ref={closeRef} type="button" className="btn btn-secondary btn-small" onClick={onClose}>
            Закрыть <span className="kbd">Esc</span>
          </button>
        </div>
        <img className="modal-img" src={src} alt={caption} />
      </div>
    </div>
  );
}
