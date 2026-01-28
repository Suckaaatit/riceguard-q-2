package com.ximbizz.qualityai;

import android.app.Activity;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.TextView;

import com.google.ar.core.Config;
import com.google.ar.core.Frame;
import com.google.ar.core.HitResult;
import com.google.ar.core.Plane;
import com.google.ar.core.Pose;
import com.google.ar.core.Session;
import com.google.ar.core.Trackable;
import com.google.ar.core.TrackingState;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.List;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public class RiceARActivity extends Activity implements GLSurfaceView.Renderer {

    private GLSurfaceView glView;
    private TextView overlay;

    private Session session;
    private int cameraTextureId = -1;

    private int program = 0;
    private int aPosLoc = -1;
    private int aUvLoc = -1;
    private int uTexLoc = -1;

    private FloatBuffer quadPos;
    private FloatBuffer quadUv;
    private FloatBuffer uvIn;
    private FloatBuffer uvOut;

    private int viewW = 0;
    private int viewH = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        RiceARState.setActivity(this);
        RiceARState.setError("");
        RiceARState.setGuidance("");
        RiceARState.setDistanceCm(null);
        RiceARState.setRunning(true);

        FrameLayout root = new FrameLayout(this);
        glView = new GLSurfaceView(this);
        glView.setEGLContextClientVersion(2);
        glView.setRenderer(this);
        glView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);

        overlay = new TextView(this);
        overlay.setTextSize(18f);
        overlay.setPadding(24, 24, 24, 24);
        overlay.setGravity(Gravity.CENTER);

        FrameLayout.LayoutParams lp = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        lp.gravity = Gravity.TOP;

        root.addView(glView);
        root.addView(overlay, lp);
        setContentView(root);
    }

    @Override
    protected void onResume() {
        super.onResume();
        try {
            if (session == null) {
                session = new Session(this);
                Config config = new Config(session);
                config.setUpdateMode(Config.UpdateMode.LATEST_CAMERA_IMAGE);
                session.configure(config);
            }
            if (cameraTextureId != -1) {
                session.setCameraTextureName(cameraTextureId);
            }
            session.resume();
            glView.onResume();
        } catch (Exception e) {
            RiceARState.setError(e.getMessage());
            RiceARState.setRunning(false);
            finish();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        try {
            glView.onPause();
            if (session != null) {
                session.pause();
            }
        } catch (Exception ignored) {
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        RiceARState.setRunning(false);
        RiceARState.clearActivity(this);
        try {
            if (session != null) {
                session.close();
                session = null;
            }
        } catch (Exception ignored) {
        }
    }

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        cameraTextureId = createExternalTexture();

        String vs = "attribute vec4 a_Position;\n" +
                "attribute vec2 a_TexCoord;\n" +
                "varying vec2 v_TexCoord;\n" +
                "void main() {\n" +
                "  gl_Position = a_Position;\n" +
                "  v_TexCoord = a_TexCoord;\n" +
                "}\n";

        String fs = "#extension GL_OES_EGL_image_external : require\n" +
                "precision mediump float;\n" +
                "varying vec2 v_TexCoord;\n" +
                "uniform samplerExternalOES sTexture;\n" +
                "void main() {\n" +
                "  gl_FragColor = texture2D(sTexture, v_TexCoord);\n" +
                "}\n";

        program = buildProgram(vs, fs);
        aPosLoc = GLES20.glGetAttribLocation(program, "a_Position");
        aUvLoc = GLES20.glGetAttribLocation(program, "a_TexCoord");
        uTexLoc = GLES20.glGetUniformLocation(program, "sTexture");

        quadPos = ByteBuffer.allocateDirect(8 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        quadPos.put(new float[]{-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f});
        quadPos.position(0);

        quadUv = ByteBuffer.allocateDirect(8 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        quadUv.put(new float[]{0f, 1f, 1f, 1f, 0f, 0f, 1f, 0f});
        quadUv.position(0);

        uvIn = ByteBuffer.allocateDirect(8 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        uvIn.put(new float[]{0f, 0f, 1f, 0f, 0f, 1f, 1f, 1f});
        uvIn.position(0);
        uvOut = ByteBuffer.allocateDirect(8 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        uvOut.position(0);

        try {
            if (session != null) {
                session.setCameraTextureName(cameraTextureId);
            }
        } catch (Exception e) {
            RiceARState.setError(e.getMessage());
        }

        GLES20.glClearColor(0f, 0f, 0f, 1f);
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        viewW = width;
        viewH = height;
        GLES20.glViewport(0, 0, width, height);
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);

        if (session == null) {
            return;
        }

        try {
            Frame frame = session.update();

            try {
                uvIn.position(0);
                uvOut.position(0);
                frame.transformDisplayUvCoords(uvIn, uvOut);
                uvOut.position(0);
                quadUv.position(0);
                quadUv.put(uvOut);
                quadUv.position(0);
            } catch (Exception ignored) {
            }

            drawCameraBackground();

            if (frame.getCamera().getTrackingState() != TrackingState.TRACKING) {
                setOverlayText("Searching for surface...");
                RiceARState.setGuidance("Searching for surface...");
                RiceARState.setDistanceCm(null);
                return;
            }

            float cx = viewW / 2f;
            float cy = viewH / 2f;
            List<HitResult> hits = frame.hitTest(cx, cy);
            Float bestCm = null;

            for (HitResult hit : hits) {
                Trackable t = hit.getTrackable();
                if (t instanceof Plane) {
                    Plane plane = (Plane) t;
                    Pose pose = hit.getHitPose();
                    if (!plane.isPoseInPolygon(pose)) continue;

                    Pose cam = frame.getCamera().getPose();
                    float dx = cam.tx() - pose.tx();
                    float dy = cam.ty() - pose.ty();
                    float dz = cam.tz() - pose.tz();
                    float dist = (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
                    bestCm = dist * 100f;
                    break;
                }
            }

            if (bestCm == null) {
                setOverlayText("No plane detected");
                RiceARState.setGuidance("No plane detected");
                RiceARState.setDistanceCm(null);
                return;
            }

            RiceARState.setDistanceCm(bestCm);

            String g;
            if (bestCm < 14f) g = "Move back";
            else if (bestCm > 16f) g = "Move closer";
            else g = "Perfect distance";

            RiceARState.setGuidance(g);
            setOverlayText(g + " (" + Math.round(bestCm) + "cm)");

        } catch (Exception e) {
            RiceARState.setError(e.getMessage());
            RiceARState.setGuidance("AR error");
            RiceARState.setDistanceCm(null);
            setOverlayText("AR error");
        }
    }

    private void setOverlayText(final String text) {
        runOnUiThread(() -> overlay.setText(text));
    }

    private int createExternalTexture() {
        int[] tex = new int[1];
        GLES20.glGenTextures(1, tex, 0);
        int id = tex[0];
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, id);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        return id;
    }

    private int buildProgram(String vs, String fs) {
        int v = compileShader(GLES20.GL_VERTEX_SHADER, vs);
        int f = compileShader(GLES20.GL_FRAGMENT_SHADER, fs);
        int p = GLES20.glCreateProgram();
        GLES20.glAttachShader(p, v);
        GLES20.glAttachShader(p, f);
        GLES20.glLinkProgram(p);
        return p;
    }

    private int compileShader(int type, String src) {
        int s = GLES20.glCreateShader(type);
        GLES20.glShaderSource(s, src);
        GLES20.glCompileShader(s);
        return s;
    }

    private void drawCameraBackground() {
        if (program == 0) return;

        GLES20.glUseProgram(program);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTextureId);
        GLES20.glUniform1i(uTexLoc, 0);

        quadPos.position(0);
        GLES20.glEnableVertexAttribArray(aPosLoc);
        GLES20.glVertexAttribPointer(aPosLoc, 2, GLES20.GL_FLOAT, false, 0, quadPos);

        quadUv.position(0);
        GLES20.glEnableVertexAttribArray(aUvLoc);
        GLES20.glVertexAttribPointer(aUvLoc, 2, GLES20.GL_FLOAT, false, 0, quadUv);

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

        GLES20.glDisableVertexAttribArray(aPosLoc);
        GLES20.glDisableVertexAttribArray(aUvLoc);
    }
}
