package com.ximbizz.qualityai // ⚠️ CHECK YOUR PACKAGE NAME

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Color
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.camera2.CaptureResult
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.getcapacitor.JSObject
import com.getcapacitor.Plugin
import com.getcapacitor.PluginCall
import com.getcapacitor.PluginMethod
import com.getcapacitor.annotation.CapacitorPlugin
import com.getcapacitor.annotation.Permission
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs

@CapacitorPlugin(
    name = "RiceGuardCamera",
    permissions = [
        Permission(alias = "camera", strings = [Manifest.permission.CAMERA]),
        Permission(alias = "storage", strings = [Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE])
    ]
)
class RiceGuardCameraPlugin : Plugin(), SensorEventListener {

    private lateinit var cameraExecutor: ExecutorService
    private var previewView: PreviewView? = null
    private var imageCapture: ImageCapture? = null
    private var camera: Camera? = null
    
    // SENSORS
    private var sensorManager: SensorManager? = null
    private var gyroscope: Sensor? = null
    private var linearAccel: Sensor? = null
    private var vibrator: Vibrator? = null

    // STATE
    private var isTiltSafe = false
    private var lastVibrateTime: Long = 0

    override fun load() {
        super.load()
        cameraExecutor = Executors.newSingleThreadExecutor()
        sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        
        // 1. Gyroscope for TILT (keeping phone flat)
        gyroscope = sensorManager?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        
        // 2. Linear Acceleration for UP/DOWN MOVEMENT (Z-Axis)
        linearAccel = sensorManager?.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
        
        // 3. Vibrator Service
        vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vibratorManager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
            vibratorManager.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }
    }

    @PluginMethod
    fun startCamera(call: PluginCall) {
        if (!hasRequiredPermissions()) {
            requestPermissions(call)
            return
        }

        activity.runOnUiThread {
            if (previewView == null) {
                previewView = PreviewView(context).apply {
                    layoutParams = FrameLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    )
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                }
                val rootView = bridge.webView.parent as ViewGroup
                rootView.addView(previewView, 0)
                bridge.webView.setBackgroundColor(Color.TRANSPARENT)
            }

            bindCameraUseCases()
            
            // REGISTER SENSORS
            gyroscope?.let { sensorManager?.registerListener(this, it, SensorManager.SENSOR_DELAY_UI) }
            linearAccel?.let { sensorManager?.registerListener(this, it, SensorManager.SENSOR_DELAY_UI) }
            
            call.resolve()
        }
    }

    @PluginMethod
    fun stopCamera(call: PluginCall) {
        activity.runOnUiThread {
            val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
            cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()
                cameraProvider.unbindAll()
                previewView?.let {
                    val rootView = bridge.webView.parent as ViewGroup
                    rootView.removeView(it)
                    previewView = null
                }
                bridge.webView.setBackgroundColor(Color.WHITE)
                sensorManager?.unregisterListener(this)
                call.resolve()
            }, ContextCompat.getMainExecutor(context))
        }
    }

    @PluginMethod
    fun capturePhoto(call: PluginCall) {
        if (!isTiltSafe) {
            call.reject("Tilt Error: Keep phone flat.")
            vibrateNative(500) // Long vibrate on error
            return
        }

        val imageCapture = imageCapture ?: run {
            call.reject("Camera not initialized")
            return
        }

        val photoFile = File(context.cacheDir, "rice_capture_${System.currentTimeMillis()}.jpg")
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    call.reject("Capture failed: ${exc.message}")
                }
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    val ret = JSObject()
                    ret.put("path", photoFile.absolutePath)
                    call.resolve(ret)
                    vibrateNative(50) // Success click
                }
            }
        )
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build()
            preview.setSurfaceProvider(previewView?.surfaceProvider)
            imageCapture = ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY).build()
            
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            try {
                cameraProvider.unbindAll()
                camera = cameraProvider.bindToLifecycle(bridge.activity, cameraSelector, preview, imageCapture)
            } catch (exc: Exception) {
                Log.e("RiceGuard", "Binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    // --- SENSOR LOGIC ---
    override fun onSensorChanged(event: SensorEvent?) {
        event ?: return

        // 1. GYROSCOPE (TILT CHECK)
        if (event.sensor.type == Sensor.TYPE_GYROSCOPE) {
            val x = event.values[0]
            val y = event.values[1]
            // Tilt Limit: ~5 degrees (0.1 rad)
            isTiltSafe = (abs(x) < 0.1 && abs(y) < 0.1)
            
            // Notify Frontend of Tilt State
            val data = JSObject()
            data.put("isTiltSafe", isTiltSafe)
            notifyListeners("guardrailState", data)
        }

        // 2. LINEAR ACCELERATION (UP/DOWN MOVEMENT)
        // 
        // Z-Axis (Index 2) is perpendicular to the screen.
        // When holding phone flat, Z-axis movement = moving closer/further from rice.
        if (event.sensor.type == Sensor.TYPE_LINEAR_ACCELERATION) {
            val zAccel = event.values[2] 

            // Threshold: 3.0 m/s^2 (Significant movement)
            if (abs(zAccel) > 3.0) {
                // Throttle vibration (don't buzz continuously)
                val now = System.currentTimeMillis()
                if (now - lastVibrateTime > 300) {
                    vibrateNative(20) // Tiny "Tick" feedback
                    lastVibrateTime = now
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun vibrateNative(duration: Long) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator?.vibrate(VibrationEffect.createOneShot(duration, VibrationEffect.DEFAULT_AMPLITUDE))
        } else {
            @Suppress("DEPRECATION")
            vibrator?.vibrate(duration)
        }
    }
}