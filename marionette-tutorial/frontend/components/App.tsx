/**
 * Marionette 前端可视化组件
 *
 * 使用 React + Deck.gl 实现轨迹数据的双屏可视化
 */

import React, { useState, useCallback } from 'react';
import DeckGL from '@deck.gl/react';
import { ScatterplotLayer, PathLayer } from '@deck.gl/layers';
import { ViewState } from '@deck.gl/core';

// ============================================================================
// 数据类型定义
// ============================================================================

interface Coordinate {
  lat: number;
  lon: number;
}

interface TrajectoryPoint {
  time: number;
  poi_id: number;
  category: number;
  coordinate?: Coordinate;
}

interface GeneratedTrajectory {
  sequence_id: string;
  points: TrajectoryPoint[];
  conditions: Record<string, any>;
}

interface DualMapViewProps {
  realData: GeneratedTrajectory[];
  generatedData: GeneratedTrajectory[];
}

// ============================================================================
// 主组件: 双屏地图视图
// ============================================================================

/**
 * DualMapView 组件
 *
 * 实现左屏真实数据和右屏生成数据的同步可视化
 */
export const DualMapView: React.FC<DualMapViewProps> = ({ realData, generatedData }) => {
  // 共享的视图状态
  const [viewState, setViewState] = useState<ViewState>({
    longitude: 28.9784,
    latitude: 41.0082,
    zoom: 12,
    pitch: 0,
    bearing: 0
  });

  /**
   * 视图状态变化处理
   * 当任一屏幕的视图状态改变时，同步更新到另一屏幕
   */
  const onViewStateChange = useCallback(({ viewState: newViewState }: { viewState: ViewState }) => {
    setViewState(newViewState);
  }, []);

  /**
   * 准备路径数据
   * 将轨迹点转换为 Deck.gl PathLayer 需要的格式
   */
  const preparePathData = (trajectories: GeneratedTrajectory[]) => {
    return trajectories.flatMap(seq => {
      if (!seq.points || seq.points.length < 2) return [];

      const path = seq.points
        .filter(p => p.coordinate)
        .map(p => [p.coordinate!.lon, p.coordinate!.lat]);

      if (path.length < 2) return [];

      return [{
        path: path,
        color: [0, 128, 255, 200]
      }];
    });
  };

  /**
   * 准备散点数据
   * 将轨迹点转换为 Deck.gl ScatterplotLayer 需要的格式
   */
  const prepareScatterData = (trajectories: GeneratedTrajectory[]) => {
    return trajectories.flatMap(seq => {
      return seq.points
        .filter(p => p.coordinate)
        .map(p => ({
          coordinates: [p.coordinate!.lon, p.coordinate!.lat],
          color: [0, 128, 255, 200]
        }));
    });
  };

  // 左屏：真实数据图层
  const leftLayers = [
    // 路径图层
    new PathLayer({
      id: 'real-paths',
      data: preparePathData(realData),
      getPath: (d: any) => d.path,
      getColor: (d: any) => d.color,
      widthMinPixels: 2,
      widthMaxPixels: 2,
      rounded: true,
      opacity: 0.6
    }),
    // 散点图层
    new ScatterplotLayer({
      id: 'real-scatter',
      data: prepareScatterData(realData),
      getPosition: (d: any) => d.coordinates,
      getFillColor: [0, 128, 255],
      radiusUnits: 'pixels',
      getRadius: 6,
      opacity: 0.8,
      pickable: true
    })
  ];

  // 右屏：生成数据图层
  const rightLayers = [
    // 路径图层
    new PathLayer({
      id: 'generated-paths',
      data: preparePathData(generatedData),
      getPath: (d: any) => d.path,
      getColor: (d: any) => [255, 100, 0, 200],
      widthMinPixels: 2,
      widthMaxPixels: 2,
      rounded: true,
      opacity: 0.6
    }),
    // 散点图层
    new ScatterplotLayer({
      id: 'generated-scatter',
      data: prepareScatterData(generatedData),
      getPosition: (d: any) => d.coordinates,
      getFillColor: [255, 100, 0],
      radiusUnits: 'pixels',
      getRadius: 6,
      opacity: 0.8,
      pickable: true
    })
  ];

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100%' }}>
      {/* 左屏：真实数据 */}
      <div style={{ flex: 1, borderRight: '2px solid #ccc' }}>
        <div style={{
          position: 'absolute',
          top: '10px',
          left: '10px',
          zIndex: 1,
          background: 'white',
          padding: '8px',
          borderRadius: '4px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
        }}>
          <h3 style={{ margin: 0 }}>真实数据</h3>
          <p style={{ margin: '4px 0 0 0', fontSize: '14px' }}>
            {realData.length} 条轨迹
          </p>
        </div>
        <DeckGL
          viewState={viewState}
          onViewStateChange={onViewStateChange}
          layers={leftLayers}
          controller={true}
          getTooltip={({ object }) => {
            if (object && object.coordinates) {
              return {
                html: `
                  <div style="padding: 8px;">
                    <strong>POI ID:</strong> ${object.poi_id || 'N/A'}<br/>
                    <strong>类别:</strong> ${object.category || 'N/A'}<br/>
                    <strong>时间:</strong> ${object.time || 'N/A'}
                  </div>
                `,
                style: {
                  backgroundColor: 'white',
                  border: '1px solid #ccc',
                  borderRadius: '4px',
                  padding: '8px'
                }
              };
            }
            return null;
          }}
        />
      </div>

      {/* 右屏：生成数据 */}
      <div style={{ flex: 1 }}>
        <div style={{
          position: 'absolute',
          top: '10px',
          right: '10px',
          zIndex: 1,
          background: 'white',
          padding: '8px',
          borderRadius: '4px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
        }}>
          <h3 style={{ margin: 0 }}>生成数据</h3>
          <p style={{ margin: '4px 0 0 0', fontSize: '14px' }}>
            {generatedData.length} 条轨迹
          </p>
        </div>
        <DeckGL
          viewState={viewState}
          onViewStateChange={onViewStateChange}
          layers={rightLayers}
          controller={true}
          getTooltip={({ object }) => {
            if (object && object.coordinates) {
              return {
                html: `
                  <div style="padding: 8px;">
                    <strong>POI ID:</strong> ${object.poi_id || 'N/A'}<br/>
                    <strong>类别:</strong> ${object.category || 'N/A'}<br/>
                    <strong>时间:</strong> ${object.time || 'N/A'}
                  </div>
                `,
                style: {
                  backgroundColor: 'white',
                  border: '1px solid #ccc',
                  borderRadius: '4px',
                  padding: '8px'
                }
              };
            }
            return null;
          }}
        />
      </div>
    </div>
  );
};

// ============================================================================
// 控制面板组件
// ============================================================================

interface ControlPanelProps {
  onGenerate: (params: GenerateParams) => void;
  loading: boolean;
}

interface GenerateParams {
  dayOfWeek: number;
  isRamadan: boolean;
  isHoliday: boolean;
  numSamples: number;
}

/**
 * ControlPanel 组件
 *
 * 提供参数配置和生成控制的用户界面
 */
export const ControlPanel: React.FC<ControlPanelProps> = ({ onGenerate, loading }) => {
  const [params, setParams] = useState<GenerateParams>({
    dayOfWeek: 25,
    isRamadan: false,
    isHoliday: false,
    numSamples: 10
  });

  const dayOptions = [
    { value: 25, label: '周一' },
    { value: 26, label: '周二' },
    { value: 27, label: '周三' },
    { value: 28, label: '周四' },
    { value: 29, label: '周五' },
    { value: 30, label: '周六' },
    { value: 31, label: '周日' }
  ];

  const handleGenerate = () => {
    onGenerate(params);
  };

  return (
    <div style={{
      position: 'absolute',
      top: '10px',
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: 1000,
      background: 'white',
      padding: '16px',
      borderRadius: '8px',
      boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
      minWidth: '300px'
    }}>
      <h3 style={{ margin: '0 0 16px 0' }}>生成条件</h3>

      {/* 星期几选择 */}
      <div style={{ marginBottom: '12px' }}>
        <label style={{ display: 'block', marginBottom: '4px', fontWeight: 'bold' }}>
          星期几:
        </label>
        <select
          value={params.dayOfWeek}
          onChange={(e) => setParams({ ...params, dayOfWeek: parseInt(e.target.value) })}
          style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
        >
          {dayOptions.map(opt => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </div>

      {/* 斋月开关 */}
      <div style={{ marginBottom: '12px' }}>
        <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={params.isRamadan}
            onChange={(e) => setParams({ ...params, isRamadan: e.target.checked })}
            style={{ marginRight: '8px' }}
          />
          斋月
        </label>
      </div>

      {/* 假期开关 */}
      <div style={{ marginBottom: '12px' }}>
        <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={params.isHoliday}
            onChange={(e) => setParams({ ...params, isHoliday: e.target.checked })}
            style={{ marginRight: '8px' }}
          />
          假期
        </label>
      </div>

      {/* 生成数量 */}
      <div style={{ marginBottom: '16px' }}>
        <label style={{ display: 'block', marginBottom: '4px', fontWeight: 'bold' }}>
          生成数量: {params.numSamples}
        </label>
        <input
          type="range"
          min="1"
          max="50"
          value={params.numSamples}
          onChange={(e) => setParams({ ...params, numSamples: parseInt(e.target.value) })}
          style={{ width: '100%' }}
        />
      </div>

      {/* 生成按钮 */}
      <button
        onClick={handleGenerate}
        disabled={loading}
        style={{
          width: '100%',
          padding: '12px',
          background: loading ? '#ccc' : '#007bff',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: loading ? 'not-allowed' : 'pointer',
          fontSize: '16px',
          fontWeight: 'bold'
        }}
      >
        {loading ? '生成中...' : '生成轨迹'}
      </button>
    </div>
  );
};

// ============================================================================
// 主应用组件
// ============================================================================

/**
 * App 组件
 *
 * 整合控制面板和双屏视图
 */
const App: React.FC = () => {
  const [realData, setRealData] = useState<GeneratedTrajectory[]>([]);
  const [generatedData, setGeneratedData] = useState<GeneratedTrajectory[]>([]);
  const [loading, setLoading] = useState(false);

  /**
   * 处理生成请求
   * 调用后端 API 生成轨迹
   */
  const handleGenerate = async (params: GenerateParams) => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/v1/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          condition1: params.dayOfWeek,
          condition2: params.isRamadan ? 33 : 32,
          condition3: params.isHoliday ? 35 : 34,
          num_samples: params.numSamples,
          tmax: 86400.0
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setGeneratedData(data.sequences);
    } catch (error) {
      console.error('生成失败:', error);
      alert('生成失败，请检查后端服务是否运行');
    } finally {
      setLoading(false);
    }
  };

  // 初始加载真实数据
  React.useEffect(() => {
    // TODO: 从后端或文件加载真实数据
    // 这里使用模拟数据
    setRealData([
      {
        sequence_id: 'real-1',
        points: [
          {
            time: 3600,
            poi_id: 10,
            category: 0,
            coordinate: { lat: 41.0082, lon: 28.9784 }
          },
          {
            time: 7200,
            poi_id: 45,
            category: 2,
            coordinate: { lat: 41.0123, lon: 28.9821 }
          }
        ],
        conditions: { day_of_week: 25 }
      }
    ]);
  }, []);

  return (
    <div style={{ position: 'relative', height: '100vh', width: '100vw' }}>
      {/* 控制面板 */}
      <ControlPanel onGenerate={handleGenerate} loading={loading} />

      {/* 双屏地图视图 */}
      <DualMapView realData={realData} generatedData={generatedData} />
    </div>
  );
};

export default App;
