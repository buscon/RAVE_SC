/*
RAVE resynthesis: a real recording (or the microphone) goes through the model's encoder, the latent vector
is changed by a value between 0 and 1 (like the proximity value in Sentire), and the decoder turns it
back into sound. With the value in the middle (0.5) and "offset" mode, nothing is changed: this is plain
resynthesis (encode, then decode), and it shows how the model sounds when it gets realistic latents.

Modes
- offset: latent + direction * (value - 0.5) * 2 * range. The direction is one latent dimension, a random
  direction, or all dimensions together. Value 0.5 = no change.
- blend A/B: the latent vectors of two inputs are mixed: value 0 = only A, value 1 = only B. Two copies of the
  model are loaded for this (one per input), because the encoder keeps internal state.
  B is the file you choose with "choose file B"; without one, B is a later part of the file A.

Other controls
- input gain and "auto level": the models only respond in a narrow level range. Auto level adjusts the input
  slowly to an rms of about 0.06. In my Python tests with the seawaves recording, an rms of 0.03 to 0.1 worked.
- latent smoothing: low-pass filter on the latent vectors (in seconds). It can reduce roughness, and it also
  slows down the response to the value.
- "mouse X" uses the mouse as the value (instead of the slider).
- "listen to": switches between the model output and the original input (same level, no model). Use it to compare
  what the model makes of A and B with the originals. At value 0 (blend mode) you hear the model's version of A,
  at value 1 its version of B, not the originals.
- the bars show the latent vector that goes into the decoder.

Requirements: SuperCollider 3.13+, nn.ar, the models folder below, and a recording (default audio/Seawaves1.wav).
I could not run SuperCollider, so this is untested. If something fails, send me the post window text.
*/

// 1. Setup (run once). The server options are taken from your startup file (48000 Hz).
(
s.waitForBoot {
	~rsDir = "/Users/marcellolussana/Documents/Software/RAVE_SC/";
	~rsModelDir = ~rsDir ++ "models/";

	~rsFiles = PathName(~rsModelDir).files
		.select { |f| f.extension == "ts" }
		.reject { |f| f.fileName.contains("r44100") }
		.sort { |a, b| a.fileName < b.fileName };

	// latent size from the file name: "..._z16.ts" or "..._18d.ts"
	~rsLatentSize = { |name|
		var m = name.findRegexp("[_-]z(\\d+)\\.ts");
		if(m.isEmpty) { m = name.findRegexp("_(\\d+)d\\.ts") };
		if(m.isEmpty) { nil } { m[1][1].asInteger };
	};

	~rsLoaded = Set.new;
	~rsIdx = 0;
	~rsZ = 8;
	~rsMode = \offset;
	~rsSrcA = \file;
	~rsSrcB = \later;
	~rsBufA = nil;
	~rsBufB = nil;
	~rsDirKind = 0;       // 0..z-1: that dimension; z: random direction; z+1: all dimensions
	~rsLatBus = ~rsLatBus ?? { Bus.control(s, 24) };

	~rsReadA = { |path|
		if(~rsBufA.notNil) { ~rsBufA.free };
		~rsBufA = Buffer.read(s, path);
	};
	~rsReadB = { |path|
		if(~rsBufB.notNil) { ~rsBufB.free };
		~rsBufB = Buffer.read(s, path);
	};
	~rsPathA = ~rsDir ++ "audio/Seawaves1.wav";
	~rsReadA.(~rsPathA);
	s.sync;

	"Models found:".postln;
	~rsFiles.do { |f, i|
		"  %: %  (latent size %)".format(i, f.fileName, ~rsLatentSize.(f.fileName)).postln;
	};
};
)

// 2. Core: direction, building the sound, loading the model
(
~rsRandDir = { |z|
	var d = Array.fill(z) { 1.0.rand2 };
	d / d.squared.mean.sqrt;   // average size 1 per dimension
};

// sets the \dir control from ~rsDirKind
~rsSetDir = {
	var z = ~rsZ;
	var d;
	if(~rsDirKind < z) {
		d = Array.fill(z) { |i| if(i == ~rsDirKind) { 1.0 } { 0.0 } };
	} {
		d = if(~rsDirKind == z) { ~rsRandDir.(z) } { Array.fill(z, 1.0) };
	};
	Ndef(\resyn).set(\dir, d);
};

// input signal: A or B. mic, file, or a later part of file A
~rsInput = { |which|
	var sig;
	if(which == \A) {
		sig = if(~rsSrcA == \mic) { SoundIn.ar(0) } {
			Mix(PlayBuf.ar(~rsBufA.numChannels, ~rsBufA, BufRateScale.kr(~rsBufA), loop: 1))
		};
	} {
		sig = case
			{ ~rsSrcB == \mic } { SoundIn.ar(0) }
			{ (~rsSrcB == \file) and: { ~rsBufB.notNil } } {
				Mix(PlayBuf.ar(~rsBufB.numChannels, ~rsBufB, BufRateScale.kr(~rsBufB), loop: 1))
			}
			{
				Mix(PlayBuf.ar(~rsBufA.numChannels, ~rsBufA, BufRateScale.kr(~rsBufA),
					startPos: 0.5 * ~rsBufA.numFrames, loop: 1))
			};
	};
	sig
};

// slow level adjustment towards an rms of about 0.06 (only when \auto is 1)
~rsLevel = { |in, inGain, auto|
	var lvl = Amplitude.kr(in, 0.5, 1).max(0.002);
	var comp = (0.06 / lvl).clip(0.05, 20);
	in * inGain * Select.kr(auto, [1, comp])
};

~rsBuild = {
	var nameA = ("m" ++ ~rsIdx).asSymbol;
	var nameB = ("b" ++ ~rsIdx).asSymbol;
	var z = ~rsZ;
	var mode = ~rsMode;
	Ndef(\resyn).fadeTime = 0.5;
	Ndef(\resyn, {
		var prox = \prox.kr(0.5);
		var range = \range.kr(3);
		var amp = \amp.kr(1);
		var inGain = \inGain.kr(1);
		var auto = \auto.kr(0);
		var smooth = \smooth.kr(0);
		var listen = \listen.kr(0);   // 0 = model output, 1 = original A, 2 = original B (same level, no model)
		var dir = \dir.kr(Array.fill(z, 0), 0.1);
		var inA = ~rsLevel.(~rsInput.(\A), inGain, auto);
		var latA = NN(nameA, \encode).ar(inA);
		var shifted, latB, inB, out;
		if(mode == \blend) {
			inB = ~rsLevel.(~rsInput.(\B), inGain, auto);
			latB = NN(nameB, \encode).ar(inB);
			shifted = latA + ((latB - latA) * prox);
		} {
			shifted = latA + (dir * (prox.linlin(0, 1, -1, 1) * range));
		};
		shifted = Lag.ar(shifted, smooth);
		out = NN(nameA, \decode).ar(shifted);
		out = Select.ar(listen, [out, inA, if(mode == \blend) { inB } { inA }]);
		Out.kr(~rsLatBus.index, A2K.kr(shifted));
		Limiter.ar(Pan2.ar(out, 0, amp), 0.8);
	});
	Ndef(\resyn).set(\range, ~rsRange ? 3, \amp, ~rsAmp ? 1, \inGain, ~rsInGain ? 1,
		\auto, ~rsAuto ? 0, \smooth, ~rsSmooth ? 0, \listen, ~rsListen ? 0);
	if(~rsMouse == true) {
		Ndef(\resyn).map(\prox, Ndef(\rsProx));
	} {
		Ndef(\resyn).unmap(\prox);
		Ndef(\resyn).set(\prox, ~rsValue ? 0.5);   // the slider sets the control of the sound directly
	};
	~rsSetDir.value;
	Ndef(\resyn).play;
};

~rsUse = { |index|
	{
		var file = ~rsFiles[index];
		var z = ~rsLatentSize.(file.fileName);
		var nameA = ("m" ++ index).asSymbol;
		var nameB = ("b" ++ index).asSymbol;
		if(z.isNil) {
			"Could not read the latent size from %, skipping".format(file.fileName).warn;
		} {
			if(~rsLoaded.includes(nameA).not) {
				NN.load(nameA, file.fullPath);
				s.sync;
				~rsLoaded.add(nameA);
			};
			if((~rsMode == \blend) and: { ~rsLoaded.includes(nameB).not }) {
				NN.load(nameB, file.fullPath);   // second copy: the encoder of A and the one of B need their own state
				s.sync;
				~rsLoaded.add(nameB);
			};
			~rsZ = z;
			~rsIdx = index;
			if(~rsDirKind > (z + 1)) { ~rsDirKind = 0 };
			Ndef(\resyn).unset(\dir);   // an old array may have another size
			~rsBuild.value;
			"Playing %: % (latent size %), mode %".format(index, file.fileName, z, ~rsMode).postln;
			{ ~rsRefreshGui.value }.defer;
		};
	}.fork;
};

// the value 0..1: slider (sets the control "prox" of Ndef(\resyn) directly) or mouse X
~rsMouse = false;
~rsUseMouse = { |on|
	~rsMouse = on;
	if(on) {
		Ndef(\rsProx, { MouseX.kr(0, 1).lag(0.1) });
		Ndef(\resyn).map(\prox, Ndef(\rsProx));
	} {
		Ndef(\resyn).unmap(\prox);
		Ndef(\resyn).set(\prox, ~rsValue ? 0.5);
	};
};
)

// 3. Window
(
if(~rsWin.notNil and: { ~rsWin.isClosed.not }) { ~rsWin.close };
~rsWin = Window("RAVE resynthesis", Rect(80, 80, 760, 640));

~rsModelPop = PopUpMenu().items_(~rsFiles.collect { |f| f.fileName }).action_({ |m| ~rsUse.(m.value) });

~rsModePop = PopUpMenu().items_(["offset (value 0.5 = unchanged)", "blend A / B (0 = A, 1 = B)"])
	.action_({ |m|
		~rsMode = if(m.value == 0) { \offset } { \blend };
		~rsUse.(~rsIdx);
	});
~rsDirPop = PopUpMenu().items_(["dim 0"]).action_({ |m| ~rsDirKind = m.value; ~rsSetDir.value });
~rsNewDirBtn = Button().states_([["new random direction"]]).action_({
	~rsDirKind = ~rsZ;
	~rsDirPop.value = ~rsZ;
	~rsSetDir.value;
});

~rsSrcAPop = PopUpMenu().items_(["A: file", "A: microphone (use headphones)"])
	.action_({ |m| ~rsSrcA = if(m.value == 0) { \file } { \mic }; ~rsBuild.value });
~rsFileA = Button().states_([["choose file A..."]]).action_({
	FileDialog({ |path|
		{
			~rsPathA = path;
			~rsReadA.(path);
			s.sync;
			~rsFileALabel.string = "A: " ++ path.basename;
			~rsBuild.value;
		}.fork(AppClock);
	}, {}, 1, 0, true);
});
~rsFileALabel = StaticText().string_("A: " ++ ~rsPathA.basename);
~rsSrcBPop = PopUpMenu().items_(["B: later part of A", "B: file", "B: microphone"])
	.action_({ |m| ~rsSrcB = [\later, \file, \mic][m.value]; ~rsBuild.value });
~rsFileB = Button().states_([["choose file B..."]]).action_({
	FileDialog({ |path|
		{
			~rsReadB.(path);
			s.sync;
			~rsFileBLabel.string = "B: " ++ path.basename;
			~rsSrcB = \file;
			~rsSrcBPop.value = 1;
			~rsBuild.value;
		}.fork(AppClock);
	}, {}, 1, 0, true);
});
~rsFileBLabel = StaticText().string_("B: later part of A");

~rsValue = 0.5;
~rsValLabel = StaticText().string_("value 0.50");
~rsValSl = Slider().orientation_(\horizontal).value_(0.5).minSize_(Size(100, 30)).action_({ |sl|
	~rsValue = sl.value;
	if(~rsMouse != true) { Ndef(\resyn).set(\prox, sl.value) };
	~rsValLabel.string = "value " ++ sl.value.round(0.01).asString;
});
~rsMouseBtn = Button().states_([["mouse X: off"], ["mouse X: on"]]).action_({ |b| ~rsUseMouse.(b.value == 1) });

~rsRange = 3; ~rsAmp = 1; ~rsInGain = 1; ~rsAuto = 0; ~rsSmooth = 0;
~rsRangeSpec = ControlSpec(0.2, 10, \exp);
~rsAmpSpec = ControlSpec(0.1, 10, \exp);
~rsInSpec = ControlSpec(0.1, 20, \exp);
~rsSmoothSpec = ControlSpec(0.001, 0.5, \exp);
~rsRangeSl = Slider().orientation_(\horizontal).value_(~rsRangeSpec.unmap(3))
	.action_({ |sl| ~rsRange = ~rsRangeSpec.map(sl.value); Ndef(\resyn).set(\range, ~rsRange) });
~rsAmpSl = Slider().orientation_(\horizontal).value_(~rsAmpSpec.unmap(1))
	.action_({ |sl| ~rsAmp = ~rsAmpSpec.map(sl.value); Ndef(\resyn).set(\amp, ~rsAmp) });
~rsInSl = Slider().orientation_(\horizontal).value_(~rsInSpec.unmap(1))
	.action_({ |sl| ~rsInGain = ~rsInSpec.map(sl.value); Ndef(\resyn).set(\inGain, ~rsInGain) });
~rsSmoothLabel = StaticText().string_("latent smoothing: off");
~rsSmoothSl = Slider().orientation_(\horizontal).value_(0).action_({ |sl|
	~rsSmooth = if(sl.value < 0.02) { 0 } { ~rsSmoothSpec.map(sl.value) };
	Ndef(\resyn).set(\smooth, ~rsSmooth);
	~rsSmoothLabel.string = if(~rsSmooth == 0) { "latent smoothing: off" } {
		"latent smoothing: % ms".format((~rsSmooth * 1000).round(1)) };
});
~rsAutoBtn = Button().states_([["auto level: off"], ["auto level: on"]]).action_({ |b|
	~rsAuto = b.value;
	Ndef(\resyn).set(\auto, ~rsAuto);
});
~rsListen = 0;
~rsListenPop = PopUpMenu().items_(["listen to: model output", "listen to: original A (no model)", "listen to: original B (no model)"])
	.action_({ |m| ~rsListen = m.value; Ndef(\resyn).set(\listen, ~rsListen) });
~rsStopBtn = Button().states_([["stop"], ["play"]]).action_({ |b|
	if(b.value == 1) { Ndef(\resyn).stop } { Ndef(\resyn).play };
});

~rsLive = MultiSliderView().size_(~rsZ).value_(Array.fill(~rsZ, 0.5))
	.isFilled_(true).elasticMode_(1).editable_(false).minSize_(Size(100, 120));

~rsWin.layout = VLayout(
	HLayout(~rsModelPop),
	HLayout(~rsSrcAPop, ~rsFileA, ~rsFileALabel),
	HLayout(~rsModePop),
	HLayout(~rsDirPop, ~rsNewDirBtn),
	HLayout(~rsSrcBPop, ~rsFileB, ~rsFileBLabel),
	HLayout(~rsValSl, ~rsValLabel, ~rsMouseBtn),
	HLayout(StaticText().string_("range (latent units, offset mode)"), ~rsRangeSl),
	HLayout(StaticText().string_("input gain"), ~rsInSl, ~rsAutoBtn),
	HLayout(~rsSmoothLabel, ~rsSmoothSl),
	HLayout(StaticText().string_("volume"), ~rsAmpSl, ~rsStopBtn),
	HLayout(~rsListenPop),
	StaticText().string_("latent vector going into the decoder (-8 .. +8)"),
	~rsLive
);

~rsRefreshGui = {
	if(~rsWin.notNil and: { ~rsWin.isClosed.not }) {
		~rsModelPop.value = ~rsIdx;
		~rsDirPop.items = (~rsZ.collect { |i| "dimension " ++ i }) ++ ["random direction", "all dimensions"];
		~rsDirPop.value = ~rsDirKind;
		~rsLive.size = ~rsZ;
		~rsLive.value = Array.fill(~rsZ, 0.5);
	};
};

// read the latent values from the server about 20 times per second
~rsPoll.stop;
~rsPoll = Routine {
	loop {
		~rsLatBus.getn(~rsZ, { |v|
			{
				if(~rsWin.notNil and: { ~rsWin.isClosed.not }) {
					~rsLive.value = v.linlin(-8, 8, 0, 1).clip(0, 1);
				};
			}.defer;
		});
		0.05.wait;
	};
}.play(AppClock);
~rsWin.onClose = { ~rsPoll.stop };
~rsWin.front;
~rsRefreshGui.value;
)

// 4. Start (index in the list printed in step 1; the water model is a good first try)
~rsUse.(~rsFiles.detectIndex { |f| f.fileName.contains("water") } ? 0);

// Then: set the value slider to 0.5 and listen: this is plain resynthesis of the recording.
// Move the value to hear the change. ~rsUse.(n) switches the model.
// From code: Ndef(\resyn).set(\prox, 0.3);   (when "mouse X" is off)
// Check that the control exists and has the value you set:
// Ndef(\resyn).controlKeys.postln; Ndef(\resyn).get(\prox).postln;

// Stop
// Ndef(\resyn).free; Ndef(\rsProx).free;
