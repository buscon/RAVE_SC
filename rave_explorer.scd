/*
RAVE explorer: 2D control (mouse or on-screen pad) of the latent space, live display of the latent
vector, and controls to move the plane around in the latent space.

How it works: shifted latent = encoder(input) + center + dirA * X * range + dirB * Y * range
- X and Y (0..1) come from the mouse or from the pad (they play the role of "proximity").
- dirA and dirB are two random directions across ALL latent dimensions (the "plane").
  "new directions" picks a new plane. This is how you reach other parts of the 8-20 dimensional space.
- center moves the origin of the plane. Drag the lower sliders (one per latent dimension), or "random center".
- The upper bars show the latent vector that goes into the decoder, updated live.
  This shows where you are right now. It is not a map of the whole space.

Requirements: SuperCollider 3.13+, nn.ar, models in the folder set below.
I could not run SuperCollider, so this is untested. If something fails, send me the post window text.
*/

// 1. Setup (run once). Prints the list of models.
(
s.options.sampleRate = 48000;   // models with r44100 in the name are skipped
s.options.numInputBusChannels = 2;
s.options.numOutputBusChannels = 2;
s.waitForBoot {
	~modelDir = "/Users/marcellolussana/Documents/Software/RAVE_SC/models/";

	~files = PathName(~modelDir).files
		.select { |f| f.extension == "ts" }
		.reject { |f| f.fileName.contains("r44100") }
		.sort { |a, b| a.fileName < b.fileName };

	// latent size from the file name: "..._z16.ts" or "..._18d.ts"
	~latentSize = { |name|
		var m = name.findRegexp("[_-]z(\\d+)\\.ts");
		if(m.isEmpty) { m = name.findRegexp("_(\\d+)d\\.ts") };
		if(m.isEmpty) { nil } { m[1][1].asInteger };
	};

	~loaded = Set.new;
	~idx = 0;
	~z = 8;
	~latBus = ~latBus ?? { Bus.control(s, 24) };   // carries the latent values to the display

	"Models found:".postln;
	~files.do { |f, i|
		"  %: %  (latent size %)".format(i, f.fileName, ~latentSize.(f.fileName)).postln;
	};
};
)

// 2. Core: controls, input, model loading
(
~range      = 4.0;   // how far X and Y push (also a slider in the window)
~centerSpan = 6.0;   // the center sliders cover -6 .. +6
~showRange  = 8.0;   // the live display covers -8 .. +8
~input      = \chirps;  // \chirps, \noise, \mic or \file (run ~use.(~idx) again after changing)
// IMPORTANT: the bird models only respond to quiet input. In my tests (dawnchorus, in Python),
// noise above rms ~0.02 decoded to silence, while chirps at rms 0.02-0.05 came through well.

// ~buf = Buffer.read(s, "/path/to/some/sound.wav");   // needed only for ~input = \file

~inputs = (
	chirps: {   // bird-like chirps at a level the model responds to
		var tr = Dust.kr(4);
		var env = Decay2.kr(tr, 0.01, 0.15);
		SinOsc.ar(TRand.kr(2500, 6000, tr) * (1 + (env * 0.3))) * env * 0.1
	},
	noise: { HPF.ar(PinkNoise.ar, 150) * LFNoise2.kr(0.4).range(0.1, 1) * 0.03 },
	mic:   { SoundIn.ar(0) },   // use headphones, otherwise feedback
	file:  { PlayBuf.ar(1, ~buf, BufRateScale.kr(~buf), loop: 1) }
);

// X and Y control signals: mouse or pad
~setControl = { |mode|
	if(mode == \mouse) {
		Ndef(\px, { MouseX.kr(0, 1).lag(0.2) });
		Ndef(\py, { MouseY.kr(1, 0).lag(0.2) });   // up = 1
	} {
		Ndef(\px, { \v.kr(0.5).lag(0.1) });
		Ndef(\py, { \v.kr(0.5).lag(0.1) });
		if(~pad.notNil) {
			Ndef(\px).set(\v, ~pad.x);
			Ndef(\py).set(\v, ~pad.y);
		};
	};
	~mode = mode;
};
~setControl.(\mouse);

// two random directions, the second made perpendicular to the first
~randDir = { |z|
	var d = Array.fill(z) { 1.0.rand2 };
	d / d.squared.mean.sqrt;   // average size 1 per dimension
};
~newPlane = {
	var a = ~randDir.(~z);
	var b = ~randDir.(~z);
	b = b - (a * ((a * b).sum / (a * a).sum));
	b = b / b.squared.mean.sqrt;
	Ndef(\rave).set(\dirA, a, \dirB, b);
};

~use = { |index|
	{
		var file = ~files[index];
		var name = ("m" ++ index).asSymbol;
		var z = ~latentSize.(file.fileName);

		if(z.isNil) {
			"Could not read the latent size from %, skipping".format(file.fileName).warn;
		} {
			if(~loaded.includes(name).not) {
				NN.load(name, file.fullPath);
				s.sync;
				~loaded.add(name);
			};
			~z = z;
			~idx = index;
			~center = Array.fill(z, 0.0);

			Ndef(\rave).unset(\dirA, \dirB, \center);   // old arrays may have another size
			Ndef(\rave).fadeTime = 1;
			Ndef(\rave, {
				var px = \px.kr(0.5);
				var py = \py.kr(0.5);
				var range = \range.kr(4);
				var amp = \amp.kr(3);
				var inGain = \inGain.kr(1);
				var dirA = \dirA.kr(Array.fill(z, 0), 0.2);
				var dirB = \dirB.kr(Array.fill(z, 0), 0.2);
				var center = \center.kr(Array.fill(z, 0), 0.2);
				var in = ~inputs[~input].value * inGain;
				var lat = NN(name, \encode).ar(in);
				var shifted = lat + center
					+ (dirA * px.linlin(0, 1, -1, 1) * range)
					+ (dirB * py.linlin(0, 1, -1, 1) * range);
				var out = NN(name, \decode).ar(shifted);
				Out.kr(~latBus.index, A2K.kr(shifted));
				Limiter.ar(Pan2.ar(out, 0, amp), 0.8);
			});
			Ndef(\rave).map(\px, Ndef(\px), \py, Ndef(\py));
			Ndef(\rave).set(\center, ~center, \range, ~range);
			~newPlane.();
			Ndef(\rave).play;
			"Playing %: % (latent size %)".format(index, file.fileName, z).postln;

			{   // update the window for the new latent size
				if(~win.notNil and: { ~win.isClosed.not }) {
					~live.size = z;
					~live.value = Array.fill(z, 0.5);
					~centerView.size = z;
					~centerView.value = Array.fill(z, 0.5);
					~popup.value = index;
				};
			}.defer;
		};
	}.fork;
};
~next = { ~use.((~idx + 1) % ~files.size) };
~prev = { ~use.((~idx - 1) % ~files.size) };
)

// 3. Window
(
if(~win.notNil and: { ~win.isClosed.not }) { ~win.close };

~win = Window("RAVE explorer", Rect(80, 80, 780, 580));

~popup = PopUpMenu().items_(~files.collect { |f| f.fileName })
	.action_({ |m| ~use.(m.value) });
~btnPrev = Button().states_([["< prev"]]).action_({ ~prev.value });
~btnNext = Button().states_([["next >"]]).action_({ ~next.value });

~pad = Slider2D().fixedWidth_(260).fixedHeight_(260).x_(0.5).y_(0.5)
	.action_({ |sl|
		if(~mode == \pad) {
			Ndef(\px).set(\v, sl.x);
			Ndef(\py).set(\v, sl.y);
		};
	});
~btnMode = Button().states_([["control: mouse"], ["control: pad"]])
	.action_({ |b| ~setControl.(if(b.value == 0) { \mouse } { \pad }) });

~live = MultiSliderView().size_(~z).value_(Array.fill(~z, 0.5))
	.isFilled_(true).elasticMode_(1).editable_(false);
~centerView = MultiSliderView().size_(~z).value_(Array.fill(~z, 0.5))
	.isFilled_(true).elasticMode_(1)
	.action_({ |v|
		~center = v.value.linlin(0, 1, ~centerSpan.neg, ~centerSpan);
		Ndef(\rave).set(\center, ~center);
	});

~btnPlane = Button().states_([["new directions"]]).action_({ ~newPlane.value });
~btnRandCenter = Button().states_([["random center"]]).action_({
	~center = Array.fill(~z) { ~centerSpan.rand2 * 0.5 };
	~centerView.value = ~center.linlin(~centerSpan.neg, ~centerSpan, 0, 1);
	Ndef(\rave).set(\center, ~center);
});
~btnResetCenter = Button().states_([["reset center"]]).action_({
	~center = Array.fill(~z, 0.0);
	~centerView.value = Array.fill(~z, 0.5);
	Ndef(\rave).set(\center, ~center);
});

~ampSpec = ControlSpec(0.3, 20, \exp);
~inSpec = ControlSpec(0.1, 20, \exp);
~rangeSpec = ControlSpec(0.5, 10, \exp);
~ampSl = Slider().orientation_(\horizontal).value_(~ampSpec.unmap(3))
	.action_({ |s| Ndef(\rave).set(\amp, ~ampSpec.map(s.value)) });
~inSl = Slider().orientation_(\horizontal).value_(~inSpec.unmap(1))
	.action_({ |s| Ndef(\rave).set(\inGain, ~inSpec.map(s.value)) });
~rangeSl = Slider().orientation_(\horizontal).value_(~rangeSpec.unmap(~range))
	.action_({ |s| ~range = ~rangeSpec.map(s.value); Ndef(\rave).set(\range, ~range) });

~win.layout = VLayout(
	HLayout(~popup, ~btnPrev, ~btnNext),
	HLayout(
		VLayout(StaticText().string_("position X / Y (pad)"), ~pad, ~btnMode),
		VLayout(
			StaticText().string_("live latent vector (after the shift)"), ~live,
			StaticText().string_("center: drag to move the origin of the plane"), ~centerView
		)
	),
	HLayout(~btnPlane, ~btnRandCenter, ~btnResetCenter),
	HLayout(
		StaticText().string_("volume"), ~ampSl,
		StaticText().string_("input gain"), ~inSl,
		StaticText().string_("range"), ~rangeSl
	)
);

// read the latent values from the server about 20 times per second
~poll.stop;
~poll = Routine {
	loop {
		~latBus.getn(~z, { |v|
			{
				if(~win.notNil and: { ~win.isClosed.not }) {
					~live.value = v.linlin(~showRange.neg, ~showRange, 0, 1).clip(0, 1);
				};
			}.defer;
		});
		0.05.wait;
	};
}.play(AppClock);
~win.onClose = { ~poll.stop };
~win.front;
)

// 4. Start
~use.(1);

// Mouse mode: move the mouse (left-right = X, up-down = Y).
// Pad mode: press "control: pad" and drag in the square.
// ~next.value; ~prev.value; ~use.(5);

// Stop
// Ndef(\rave).free; Ndef(\px).free; Ndef(\py).free;
