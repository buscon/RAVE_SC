/*
RAVE latent map explorer: click or drag on the map to play the sound at that place.

The map comes from latent_map.py (the *_map.csv file). Each dot is one latent vector with a known sound.
Colours are sound types, listed on the right with their typical brightness (spectral centroid).
Where you click, the latent vector is the weighted mean of the nearest dots, and the decoder plays it.
No input sound is needed: this plays the decoder alone, so the sound is steady. "jitter" adds slow movement.

Position on the map is a t-SNE / LDA / PCA projection, so distances are only approximate.
LDA is the most readable: brightness changes mainly along its horizontal axis.

Requirements: SuperCollider 3.13+, nn.ar, and the CSV made by latent_map.py for the same model.
I could not run SuperCollider, so this is untested. If something fails, send me the post window text.
*/

// 1. Load model and map (run once, wait for "map ready")
(
s.options.sampleRate = 48000;
s.options.numInputBusChannels = 2;
s.options.numOutputBusChannels = 2;
s.waitForBoot {
	~modelPath = "/Users/marcellolussana/Documents/Software/RAVE_SC/models/birds_dawnchorus_b2048_r48000_z8.ts";
	~mapPath   = "/Users/marcellolussana/Documents/Software/RAVE_SC/latent_maps/birds_dawnchorus_b2048_r48000_z8_decoder_map.csv";

	NN.load(\rave, ~modelPath);
	s.sync;

	// columns: x,y (t-SNE), x,y (LDA), x,y (PCA), group, centroid Hz, level dB, then the latent vector
	~rows = CSVFileReader.read(~mapPath, true, true).collect { |r| r.collect { |x| x.asFloat } };
	~nd    = ~rows[0].size - 9;
	~zAll = ~rows.collect { |r| r.drop(9) };
	~grp  = ~rows.collect { |r| r[6].asInteger };
	~cen  = ~rows.collect { |r| r[7] };
	~lvl  = ~rows.collect { |r| r[8] };
	~zMean = ~zAll.flop.collect { |c| c.mean };
	~zStd  = ~zAll.flop.collect { |c| var m = c.mean; (c - m).squared.mean.sqrt };

	~gnames = Dictionary.new;
	File.readAllString(~mapPath.replace("_map.csv", "_map_groups.txt")).split($\n).do { |l|
		var p = l.split($|);
		if(p.size == 2) { ~gnames[p[0].asInteger] = p[1] };
	};

	~byGroup = Dictionary.new;
	~grp.do { |g, i| ~byGroup[g] = (~byGroup[g] ? []).add(i) };

	~pal = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
		"#e377c2", "#7f7f7f", "#bcbd22", "#17becf"].collect { |h| Color.fromHexString(h) };
	~colorOf = { |g| if(g < 0) { Color.gray(0.75) } { ~pal[g % 10] } };

	// which projection to draw: 0 = t-SNE, 1 = LDA, 2 = PCA
	~setView = { |i|
		var c = i * 2;
		var xs = ~rows.collect { |r| r[c] };
		var ys = ~rows.collect { |r| r[c + 1] };
		~xs = xs.linlin(xs.minItem, xs.maxItem, 0.03, 0.97);
		~ys = ys.linlin(ys.minItem, ys.maxItem, 0.03, 0.97);
		~cur = nil;
		if(~uv.notNil) { ~uv.refresh };
	};
	~setView.(1);   // start with LDA

	// the decoder alone, fed with the chosen latent vector
	Ndef(\rave).fadeTime = 0.3;
	Ndef(\rave, {
		var z = \z.kr(~zMean, 0.1);
		var wob = LFNoise2.kr(2 ! ~nd) * ~zStd * \jitter.kr(0.1);
		var out = NN(\rave, \decode).ar(K2A.ar(z + wob));
		Limiter.ar(Pan2.ar(out, 0, \amp.kr(5)), 0.8);
	});

	~k = 6;          // number of neighbouring dots that are blended
	~hover = false;  // true: play while the mouse moves, without pressing

	// mx, my are 0..1 inside the map (y up)
	~go = { |mx, my|
		var d2, idx, w, wn, zt, near, info;
		mx = mx.clip(0, 1);
		my = my.clip(0, 1);
		d2 = (~xs - mx).squared + (~ys - my).squared;
		idx = d2.order.keep(~k);
		w = 1 / (d2.at(idx) + 1e-4);
		wn = w / w.sum;
		zt = Array.fill(~nd, 0);
		idx.do { |i, n| zt = zt + (~zAll[i] * wn[n]) };
		Ndef(\rave).set(\z, zt);
		near = idx[0];
		~cur = (mx: mx, my: my, idx: idx);
		info = "nearest dot: %".format(~gnames[~grp[near]] ? "silent");
		if(~cen[near] > 0) { info = info ++ "  |  centroid % Hz, level % dB".format(~cen[near].round(1), ~lvl[near].round(1)) };
		~label.string = info;
		~uv.refresh;
	};

	"map ready: % points, latent size %".format(~rows.size, ~nd).postln;
};
)

// 2. Window
(
if(~win.notNil and: { ~win.isClosed.not }) { ~win.close };

~uv = UserView().minSize_(Size(560, 520)).background_(Color.white).acceptsMouseOver_(true);
~uv.drawFunc_({ |v|
	var w = v.bounds.width, h = v.bounds.height;
	~byGroup.keysValuesDo { |g, ids|
		var c = ~colorOf.(g);
		Pen.fillColor = Color(c.red, c.green, c.blue, 0.75);
		ids.do { |i| Pen.addOval(Rect(~xs[i] * w - 2.5, (1 - ~ys[i]) * h - 2.5, 5, 5)) };
		Pen.fill;
	};
	if(~cur.notNil) {
		Pen.strokeColor = Color.black;
		Pen.width = 1;
		~cur[\idx].do { |i|
			Pen.line(Point(~cur[\mx] * w, (1 - ~cur[\my]) * h), Point(~xs[i] * w, (1 - ~ys[i]) * h));
		};
		Pen.stroke;
		Pen.width = 2;
		Pen.strokeOval(Rect(~cur[\mx] * w - 7, (1 - ~cur[\my]) * h - 7, 14, 14));
	};
});
~uv.mouseDownAction = { |v, x, y| ~go.(x / v.bounds.width, 1 - (y / v.bounds.height)) };
~uv.mouseMoveAction = { |v, x, y| ~go.(x / v.bounds.width, 1 - (y / v.bounds.height)) };
~uv.mouseOverAction = { |v, x, y|
	if(~hover) { ~go.(x / v.bounds.width, 1 - (y / v.bounds.height)) };
};

~label = StaticText().string_("click or drag on the map");

~viewPop = PopUpMenu().items_(["t-SNE", "LDA (start here)", "PCA"]).value_(1)
	.action_({ |m| ~setView.(m.value) });
~hoverBtn = Button().states_([["play on hover: off"], ["play on hover: on"]])
	.action_({ |b| ~hover = (b.value == 1) });
~kBox = NumberBox().value_(6).clipLo_(1).clipHi_(30).decimals_(0)
	.action_({ |n| ~k = n.value.asInteger });
~jitSl = Slider().orientation_(\horizontal).value_(0.1)
	.action_({ |sl| Ndef(\rave).set(\jitter, sl.value) });
~ampSpec = ControlSpec(0.5, 40, \exp);
~ampSl = Slider().orientation_(\horizontal).value_(~ampSpec.unmap(5))
	.action_({ |sl| Ndef(\rave).set(\amp, ~ampSpec.map(sl.value)) });

~legend = (~gnames.keys.asArray.sort).collect { |g|
	StaticText().string_(~gnames[g]).stringColor_(~colorOf.(g))
};

~win = Window("RAVE latent map", Rect(60, 60, 940, 640));
~win.layout = HLayout(
	[VLayout(~uv, ~label), stretch: 3],
	[VLayout(
		StaticText().string_("projection"), ~viewPop, ~hoverBtn,
		StaticText().string_("neighbours blended"), ~kBox,
		StaticText().string_("jitter (slow movement)"), ~jitSl,
		StaticText().string_("volume"), ~ampSl,
		StaticText().string_("sound types (colour: typical brightness)"),
		*~legend
	), stretch: 1]
);
~win.front;
)

// 3. Play (then click or drag in the map)
Ndef(\rave).play;

// Stop
// Ndef(\rave).free;
