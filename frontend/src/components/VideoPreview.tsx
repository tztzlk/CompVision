import { forwardRef, useImperativeHandle, useRef } from "react";

export interface VideoPreviewHandle {
  seek: (sec: number) => void;
}

interface Props {
  src: string;
  /** Браузер не смог воспроизвести файл — превью скрывается. */
  onUnavailable: () => void;
  /** Есть ли таймкоды для перехода (подсказка в подписи). */
  withTimecodes: boolean;
}

/** Локальное превью загруженного файла (URL.createObjectURL) — на сервер ничего не уходит. */
export const VideoPreview = forwardRef<VideoPreviewHandle, Props>(function VideoPreview({ src, onUnavailable, withTimecodes }, ref) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useImperativeHandle(
    ref,
    () => ({
      seek(sec: number) {
        const v = videoRef.current;
        if (!v) return;
        try {
          v.currentTime = Math.max(0, sec);
          v.pause();
          v.scrollIntoView({ behavior: "smooth", block: "nearest" });
          v.focus({ preventScroll: true });
        } catch {
          /* не критично */
        }
      },
    }),
    [],
  );

  return (
    <figure className="preview">
      <video
        ref={videoRef}
        className="preview-video"
        src={src}
        controls
        preload="metadata"
        playsInline
        muted
        onError={onUnavailable}
        onLoadedMetadata={(e) => {
          // Контейнер прочитан, но видеодорожку браузер не декодирует (например, HEVC).
          if (e.currentTarget.videoWidth === 0) onUnavailable();
        }}
        aria-label="Превью загруженного видео"
      />
      <figcaption className="hint">
        Превью загруженного файла{withTimecodes ? ". Нажмите на таймкод нарушения, чтобы перейти к моменту." : "."}
      </figcaption>
    </figure>
  );
});
